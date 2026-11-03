"""Strict DINOv3 feature extraction for the OCTA video task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from dataset_v1 import VIDEO_EXTENSIONS, discover_videos, get_animal_id, load_config, save_manifest


PREPROCESSING_VERSION = "v2_euclidean_ecc_aspect_pad_masked_regions"


def frame_quality_passes(frame_bgr: np.ndarray) -> bool:
    """Deterministic quality rule using only the current frame."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[0] < 16 or frame_bgr.shape[1] < 16:
        return False
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if not np.isfinite(gray).all():
        return False
    mean_value = float(gray.mean())
    std_value = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return 2.0 <= mean_value <= 253.0 and std_value >= 3.0 and sharpness >= 1.0


def resize_with_padding(frame_bgr: np.ndarray, image_size: int):
    """Preserve aspect ratio and return padded image plus valid-view mask."""
    height, width = frame_bgr.shape[:2]
    scale = min(image_size / max(width, 1), image_size / max(height, 1))
    new_width = max(1, min(image_size, int(round(width * scale))))
    new_height = max(1, min(image_size, int(round(height * scale))))
    resized = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    view_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    offset_x = (image_size - new_width) // 2
    offset_y = (image_size - new_height) // 2
    canvas[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized
    view_mask[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = 255
    return canvas, view_mask


class DINOv3Encoder:
    def __init__(self, model_name: str, image_size: int, device: torch.device):
        self.device = device
        self.image_size = int(image_size)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        self.hidden_size = int(
            getattr(self.model.config, "hidden_size", getattr(self.model.config, "embed_dim", 384))
        )

    @torch.inference_mode()
    def encode(self, frames_bgr, valid_masks):
        images = [Image.fromarray(frame[:, :, ::-1].copy()).convert("RGB") for frame in frames_bgr]
        try:
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                size={"height": self.image_size, "width": self.image_size},
                do_resize=True,
                do_center_crop=False,
            )
        except TypeError:
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                size={"height": self.image_size, "width": self.image_size},
            )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        tokens = outputs.last_hidden_state
        grid = self.image_size // self.patch_size
        patch_count = grid * grid
        if tokens.shape[1] < patch_count:
            raise RuntimeError(f"DINO 输出 token 数不足：{tuple(tokens.shape)}，需要至少 {patch_count} 个 patch token")

        patches = tokens[:, -patch_count:]
        patch_grid = patches.reshape(-1, grid, grid, patches.shape[-1])
        if grid % 4 != 0:
            raise ValueError(f"patch grid 必须能被4整除，当前为 {grid}×{grid}")

        block = grid // 4
        mask_grid = np.stack(
            [cv2.resize(mask, (grid, grid), interpolation=cv2.INTER_AREA) for mask in valid_masks],
            axis=0,
        ).astype(np.float32) / 255.0
        mask_grid = torch.from_numpy(mask_grid).to(device=patch_grid.device, dtype=patch_grid.dtype)

        patch_grid_6d = patch_grid.reshape(
            patch_grid.shape[0], 4, block, 4, block, patch_grid.shape[-1]
        )
        mask_regions = mask_grid.reshape(mask_grid.shape[0], 4, block, 4, block)
        weighted_sum = (patch_grid_6d * mask_regions[..., None]).sum(dim=(2, 4))
        weight_sum = mask_regions.sum(dim=(2, 4)).clamp_min(1e-6)
        regions = (weighted_sum / weight_sum[..., None]).reshape(
            patch_grid.shape[0], 16, patch_grid.shape[-1]
        )
        region_valid_mask = (weight_sum > 1e-6).reshape(patch_grid.shape[0], 16)

        if getattr(outputs, "pooler_output", None) is not None:
            global_features = outputs.pooler_output
        else:
            global_features = tokens[:, 0]

        return global_features.float().cpu(), regions.float().cpu(), region_valid_mask.cpu()


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
        raise RuntimeError(f"无法打开视频：{path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_indices(total, fps, sample_fps, max_frames)

    raw_frames = []
    timestamps = []
    for target_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ok, frame = capture.read()
        if ok and frame is not None:
            raw_frames.append(frame)
            timestamps.append(float(target_index / fps))
    capture.release()

    if not raw_frames:
        raise RuntimeError(f"视频没有可读取帧：{path}")

    reference_index = next(
        (index for index, frame in enumerate(raw_frames) if frame_quality_passes(frame)),
        0,
    )
    reference_frame = raw_frames[reference_index]
    reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    registered_frames = []
    registration_masks = []
    ecc_success = 0
    ecc_failure = 0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)

    for index, frame in enumerate(raw_frames):
        if index == reference_index:
            registered_frames.append(frame)
            registration_masks.append(np.ones(frame.shape[:2], dtype=np.uint8) * 255)
            continue

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cv2.findTransformECC(
                reference_gray,
                current_gray,
                warp,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                None,
                1,
            )
            source_valid = np.ones(frame.shape[:2], dtype=np.uint8) * 255
            aligned = cv2.warpAffine(
                frame,
                warp,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            aligned_mask = cv2.warpAffine(
                source_valid,
                warp,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            registered_frames.append(aligned)
            registration_masks.append(aligned_mask)
            ecc_success += 1
        except cv2.error:
            registered_frames.append(frame)
            registration_masks.append(np.ones(frame.shape[:2], dtype=np.uint8) * 255)
            ecc_failure += 1

    stats = {
        "fps": fps,
        "decoded_selected_frames": len(registered_frames),
        "ecc_success": ecc_success,
        "ecc_failure": ecc_failure,
        "reference_frame_index": int(reference_index),
        "reference_quality_passed": bool(frame_quality_passes(reference_frame)),
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    return registered_frames, raw_frames, registration_masks, np.asarray(timestamps, dtype=np.float32), stats


def discover_uniform_label_videos(video_root: str, label: int) -> pd.DataFrame:
    """Build a manifest when every video in one directory has a known label."""
    root = Path(video_root)
    if not root.is_dir():
        raise FileNotFoundError(f"视频目录不存在: {root}")
    if label not in (0, 1):
        raise ValueError("--all-label 必须为 0 或 1")
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"目录中没有视频: {root}")
    rows = [
        {
            "path": str(path.resolve()),
            "filename": path.name,
            "source_folder": root.name,
            "animal_id": get_animal_id(path.name),
            "label": label,
        }
        for path in paths
    ]
    manifest = pd.DataFrame(rows).sort_values(["animal_id", "filename"]).reset_index(drop=True)
    if manifest["animal_id"].duplicated().any():
        raise ValueError("目录中存在重复 animal_id，请先明确多视频聚合规则")
    manifest.insert(0, "video_id", range(len(manifest)))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v1.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-root", help="仅处理此目录中的视频；须同时指定 --all-label")
    parser.add_argument("--all-label", type=int, choices=(0, 1), help="指定目录中所有视频的统一标签")
    parser.add_argument("--manifest-path", help="本次输入视频清单的独立保存路径")
    parser.add_argument("--output-dir", help="本次特征提取审计清单的独立保存目录")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    if (args.video_root is None) != (args.all_label is None):
        parser.error("--video-root 与 --all-label 必须同时指定")
    if args.video_root:
        manifest = discover_uniform_label_videos(args.video_root, args.all_label)
        if not args.manifest_path or not args.output_dir:
            parser.error("指定 --video-root 时还须指定 --manifest-path 和 --output-dir，以免覆盖旧实验清单")
    else:
        manifest = discover_videos(config)
    if args.manifest_path:
        data_cfg["manifest_path"] = args.manifest_path
    if args.output_dir:
        data_cfg["output_dir"] = args.output_dir
    save_manifest(manifest, data_cfg["manifest_path"])

    cache_dir = Path(data_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = cache_dir / "raw_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    encoder = DINOv3Encoder(
        config["backbone"]["model_name"],
        int(data_cfg.get("image_size", 256)),
        device,
    )

    rows = []
    for row in tqdm(manifest.to_dict("records"), desc="extract_strict_v1"):
        target = cache_dir / f"{int(row['video_id']):05d}.pt"
        raw_sidecar = raw_dir / f"{int(row['video_id']):05d}.npz"

        if target.exists() and raw_sidecar.exists() and not args.overwrite:
            rows.append({
                **row,
                "cache_path": str(target.resolve()),
                "raw_frames_path": str(raw_sidecar.resolve()),
                "status": "exists",
            })
            continue

        registered_frames, raw_frames, registration_masks, timestamps, video_stats = read_registered_frames(
            row["path"],
            float(data_cfg.get("sample_fps", 2.0)),
            int(data_cfg.get("max_frames", 128)),
        )

        padded_frames = []
        view_masks = []
        for frame, registration_mask in zip(registered_frames, registration_masks):
            padded, view_mask = resize_with_padding(frame, encoder.image_size)
            registration_mask_resized = cv2.resize(
                registration_mask,
                (encoder.image_size, encoder.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            padded_frames.append(padded)
            view_masks.append(cv2.bitwise_and(view_mask, registration_mask_resized))

        global_parts = []
        region_parts = []
        region_valid_parts = []
        for start in range(0, len(padded_frames), args.batch_size):
            global_part, region_part, region_valid_part = encoder.encode(
                padded_frames[start:start + args.batch_size],
                view_masks[start:start + args.batch_size],
            )
            global_parts.append(global_part)
            region_parts.append(region_part)
            region_valid_parts.append(region_valid_part)

        np.savez_compressed(
            raw_sidecar,
            frames=np.stack(raw_frames),
            registration_masks=np.stack(registration_masks),
            view_masks=np.stack(view_masks),
        )

        payload = {
            "global_features": torch.cat(global_parts).half(),
            "region_features": torch.cat(region_parts).half(),
            "region_valid_mask": torch.cat(region_valid_parts),
            "timestamps": torch.from_numpy(timestamps),
            "video_id": int(row["video_id"]),
            "animal_id": str(row["animal_id"]),
            "label": int(row["label"]),
            "video_path": row["path"],
            "raw_frames_path": str(raw_sidecar.resolve()),
            "preprocessing_version": PREPROCESSING_VERSION,
            "video_stats": video_stats,
        }
        torch.save(payload, target)

        rows.append({
            **row,
            "cache_path": str(target.resolve()),
            "raw_frames_path": str(raw_sidecar.resolve()),
            "status": "created",
            **video_stats,
        })

    audit = Path(data_cfg["output_dir"]) / "feature_extraction_manifest.csv"
    audit.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(audit, index=False, encoding="utf-8-sig")

    print(json.dumps({
        "videos": len(rows),
        "cache_dir": str(cache_dir.resolve()),
        "audit": str(audit.resolve()),
        "preprocessing_version": PREPROCESSING_VERSION,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
