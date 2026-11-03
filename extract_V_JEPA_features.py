"""Extract frozen V-JEPA 2.1 ViT-B/16 video-clip features for B9.
python extract_V_JEPA_features.py \
    --config config_v1_V_JEPA.yaml \
    --checkpoint /home/wxy/Classification/syy/syy/strock_test/OCTA_pytorch/OCTA_VIT/weights/vjepa2_1_vitb_dist_vitG_384.pt \
    --device cuda \
    --batch-size 2 \
    --overwrite
Each model input contains 16 real consecutive sampled frames whenever possible.
Only a final short clip is padded (by repeating its last observed frame), and
the real/padded frame counts are recorded.  This padding satisfies the model's
technical input shape but never creates additional biological observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataset_v1 import discover_videos, load_config, save_manifest


PREPROCESSING_VERSION = "b9_vjepa2.1_actual_16frame_clips_384_v1"


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def frame_quality_passes(frame: np.ndarray) -> bool:
    if frame is None or frame.ndim != 3 or min(frame.shape[:2]) < 16:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return bool(
        np.isfinite(gray).all()
        and 2.0 <= float(gray.mean()) <= 253.0
        and float(gray.std()) >= 3.0
        and float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= 1.0
    )


def sample_indices(total: int, fps: float, sample_fps: float, max_frames: int):
    if total <= 0:
        return []
    stride = max(1, int(round(fps / max(sample_fps, 1e-6))))
    indices = list(range(0, total, stride))
    if max_frames > 0 and len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
        indices = [indices[int(position)] for position in positions]
    return sorted(set(indices))


def read_registered_frames(path: str, sample_fps: float, max_frames: int):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_indices(total, fps, sample_fps, max_frames)
    frames, timestamps = [], []
    for target in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(frame)
            timestamps.append(float(target / fps))
    capture.release()
    if not frames:
        raise RuntimeError(f"视频没有可读取帧: {path}")

    # The reference is selected from the opening sampled sequence by a fixed
    # quality rule. It is never selected using labels or downstream results.
    reference_index = next((i for i, f in enumerate(frames) if frame_quality_passes(f)), 0)
    reference_gray = cv2.cvtColor(frames[reference_index], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    registered, success, failure = [], 0, 0
    for i, frame in enumerate(frames):
        if i == reference_index:
            registered.append(frame)
            continue
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cv2.findTransformECC(reference_gray, current, warp, cv2.MOTION_EUCLIDEAN,
                                 criteria, None, 1)
            aligned = cv2.warpAffine(
                frame, warp, (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            registered.append(aligned)
            success += 1
        except cv2.error:
            registered.append(frame)
            failure += 1
    stats = {
        "fps": fps,
        "sample_fps": float(sample_fps),
        "decoded_selected_frames": len(registered),
        "reference_frame_index": int(reference_index),
        "reference_quality_passed": bool(frame_quality_passes(frames[reference_index])),
        "ecc_motion": "euclidean",
        "ecc_success": success,
        "ecc_failure": failure,
    }
    return registered, np.asarray(timestamps, dtype=np.float32), stats


def make_clips(frames, timestamps, clip_frames=16, clip_stride=16):
    if clip_frames != 16:
        raise ValueError("该 V-JEPA 2.1 配置要求每个输入 clip 为16帧")
    starts = list(range(0, len(frames), clip_stride))
    clips, metadata = [], []
    for start in starts:
        real_frames = frames[start:start + clip_frames]
        real_times = timestamps[start:start + clip_frames]
        if not real_frames:
            continue
        real_count = len(real_frames)
        clip = list(real_frames)
        if real_count < clip_frames:
            clip.extend([real_frames[-1]] * (clip_frames - real_count))
        clips.append(clip)
        metadata.append({
            "start_frame": int(start),
            "real_frame_count": int(real_count),
            "padded_frame_count": int(clip_frames - real_count),
            "start_time": float(real_times[0]),
            "end_time": float(real_times[-1]),
            "center_time": float(np.mean(real_times)),
        })
        if start + clip_frames >= len(frames):
            break
    return clips, metadata


class VJEPA2Encoder:
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.image_size = 384
        self.patch_size = 16
        self.num_frames = 16
        self.tubelet_size = 2
        self.grid_size = self.image_size // self.patch_size
        root = Path(__file__).resolve().parent / "vjepa2"
        if not (root / "app" / "vjepa_2_1" / "models" / "vision_transformer.py").is_file():
            raise FileNotFoundError(f"V-JEPA 2.1 源码不完整: {root}")
        if not (root / "src" / "masks" / "utils.py").is_file():
            raise FileNotFoundError(f"V-JEPA 2.1 缺少源码文件: {root / 'src/masks/utils.py'}")
        sys.path.insert(0, str(root))
        from app.vjepa_2_1.models.vision_transformer import vit_base

        self.model = vit_base(
            img_size=self.image_size, patch_size=self.patch_size,
            num_frames=self.num_frames, tubelet_size=self.tubelet_size,
            uniform_power=True, use_silu=False, wide_silu=True,
            use_sdpa=True, use_rope=True, img_temporal_dim_size=1,
            interpolate_rope=True, n_output_distillation=1,
        )
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"找不到 V-JEPA 2.1 权重: {checkpoint}")
        data = torch.load(checkpoint, map_location="cpu")
        state = data.get("ema_encoder")
        if state is None:
            raise KeyError(f"权重文件缺少 ema_encoder: {checkpoint}")
        cleaned = {}
        for key, value in state.items():
            while key.startswith("module.") or key.startswith("backbone."):
                key = key.split(".", 1)[1]
            cleaned[key] = value
        self.model.load_state_dict(cleaned, strict=True)
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1, 1)

    def _official_size_transform(self, frame):
        height, width = frame.shape[:2]
        scale = self.image_size / max(min(height, width), 1)
        new_h, new_w = max(self.image_size, int(round(height * scale))), max(self.image_size, int(round(width * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        y = (new_h - self.image_size) // 2
        x = (new_w - self.image_size) // 2
        crop = resized[y:y + self.image_size, x:x + self.image_size]
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    @torch.inference_mode()
    def encode(self, clips):
        arrays = []
        for clip in clips:
            arrays.append(np.stack([self._official_size_transform(frame) for frame in clip]))
        # B,T,H,W,C -> B,C,T,H,W
        inputs = torch.from_numpy(np.stack(arrays)).permute(0, 4, 1, 2, 3).float().to(self.device) / 255.0
        inputs = (inputs - self.mean) / self.std
        tokens = self.model(inputs)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        if isinstance(tokens, dict):
            tokens = tokens.get("x", tokens.get("last_hidden_state"))
        if tokens is None or tokens.ndim != 3:
            raise RuntimeError(f"V-JEPA 输出格式异常: {type(tokens)}")
        temporal_grid = self.num_frames // self.tubelet_size
        expected = temporal_grid * self.grid_size * self.grid_size
        if tokens.shape[1] != expected:
            raise RuntimeError(f"V-JEPA token数异常: {tuple(tokens.shape)}, 期望{expected}")
        grid = tokens.reshape(-1, temporal_grid, self.grid_size, self.grid_size, tokens.shape[-1])
        global_features = grid.mean(dim=(1, 2, 3))
        spatial = grid.mean(1)
        block = self.grid_size // 4
        regions = spatial.reshape(-1, 4, block, 4, block, tokens.shape[-1]).mean(dim=(2, 4)).reshape(-1, 16, tokens.shape[-1])
        return global_features.float().cpu(), regions.float().cpu()


def main():
    parser = argparse.ArgumentParser(description="B9 V-JEPA 2.1 frozen clip feature extraction")
    parser.add_argument("--config", default="config_v1.yaml")
    parser.add_argument("--checkpoint", default="/home/wxy/Classification/syy/syy/strock_test/OCTA_pytorch/OCTA_VIT/weights/vjepa2_1_vitb_dist_vitG_384.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--clip-stride", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.clip_stride < 1 or args.clip_stride > 16:
        raise ValueError("--clip-stride 必须在1到16之间")

    config = load_config(args.config)
    manifest = discover_videos(config)
    data = config["data"]
    save_manifest(manifest, data["manifest_path"])
    cache_dir = Path(data["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    encoder = VJEPA2Encoder(str(checkpoint), torch.device(args.device))
    rows = []
    for row in tqdm(manifest.to_dict("records"), desc="extract_b9_vjepa2"):
        target = cache_dir / f"{int(row['video_id']):05d}.pt"
        if target.exists() and not args.overwrite:
            rows.append({**row, "cache_path": str(target.resolve()), "status": "exists"})
            continue
        frames, timestamps, stats = read_registered_frames(
            row["path"], float(data.get("sample_fps", 2.0)), int(data.get("max_frames", 128))
        )
        clips, clip_meta = make_clips(frames, timestamps, 16, args.clip_stride)
        globals_, regions_ = [], []
        for start in range(0, len(clips), args.batch_size):
            global_part, region_part = encoder.encode(clips[start:start + args.batch_size])
            globals_.append(global_part); regions_.append(region_part)
        clip_times = torch.tensor([m["center_time"] for m in clip_meta], dtype=torch.float32)
        payload = {
            "global_features": torch.cat(globals_).half(),
            "region_features": torch.cat(regions_).half(),
            "timestamps": clip_times,
            "clip_metadata": clip_meta,
            "video_id": int(row["video_id"]), "animal_id": str(row["animal_id"]),
            "label": int(row["label"]), "video_path": row["path"],
            "backbone": "V-JEPA 2.1 ViT-B/16 distilled from ViT-G",
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "input_resolution": 384, "clip_frames": 16, "clip_stride": args.clip_stride,
            "preprocessing_version": PREPROCESSING_VERSION,
            "video_stats": {**stats, "clip_count": len(clips),
                            "real_frame_uses": int(sum(m["real_frame_count"] for m in clip_meta)),
                            "padded_frame_uses": int(sum(m["padded_frame_count"] for m in clip_meta))},
        }
        torch.save(payload, target)
        rows.append({**row, "cache_path": str(target.resolve()), "status": "created",
                     **payload["video_stats"], "preprocessing_version": PREPROCESSING_VERSION,
                     "checkpoint_sha256": checkpoint_hash})
    audit = Path(data["output_dir"]) / "feature_extraction_manifest_b9.csv"
    audit.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(audit, index=False, encoding="utf-8-sig")
    print(json.dumps({"videos":len(rows),"cache_dir":str(cache_dir.resolve()),
                      "audit":str(audit.resolve()),"checkpoint":str(checkpoint),
                      "checkpoint_sha256":checkpoint_hash,"preprocessing_version":PREPROCESSING_VERSION},
                     ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
