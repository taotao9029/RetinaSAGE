"""P3 DINOv3 extraction with fixed reference-frame Frangi weighting."""
# python extract_features_v2.py --config config_v1.yaml --device cpu --batch-size 8 --use-frangi-weight --overwrite 
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

try:
    from skimage.filters import frangi
except ImportError as exc:
    raise ImportError("P3 需要 scikit-image，请先安装 scikit-image。") from exc

from dataset_v1 import discover_videos, load_config, save_manifest


PREPROCESSING_VERSION = "p3_euclidean_ecc_reference_frangi_fixed_weights"


def frame_quality_passes(frame: np.ndarray) -> bool:
    if frame.ndim != 3 or frame.shape[0] < 16 or frame.shape[1] < 16:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not np.isfinite(gray).all():
        return False
    return (
        2.0 <= float(gray.mean()) <= 253.0
        and float(gray.std()) >= 3.0
        and float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= 1.0
    )


def resize_with_padding(frame: np.ndarray, image_size: int):
    height, width = frame.shape[:2]
    scale = min(image_size / max(width, 1), image_size / max(height, 1))
    new_width = max(1, min(image_size, int(round(width * scale))))
    new_height = max(1, min(image_size, int(round(height * scale))))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    offset_x = (image_size - new_width) // 2
    offset_y = (image_size - new_height) // 2
    canvas[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized
    mask[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = 255
    return canvas, mask


def make_reference_frangi_weight(
    reference_frame: np.ndarray,
    image_size: int,
    reference_mask: np.ndarray,
) -> np.ndarray:
    """Make a soft multi-scale Frangi map once from the fixed reference frame."""
    padded, padded_mask = resize_with_padding(reference_frame, image_size)
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    response = frangi(gray, sigmas=(1, 2, 3, 4), black_ridges=True)
    response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
    response = np.maximum(response, 0.0).astype(np.float32)
    response *= (padded_mask > 0).astype(np.float32)
    response *= (reference_mask > 0).astype(np.float32)
    maximum = float(response.max())
    if maximum > 0:
        response /= maximum
    return response.astype(np.float32)


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
    def encode(self, frames, valid_masks, structure_weight=None):
        images = [Image.fromarray(frame[:, :, ::-1].copy()).convert("RGB") for frame in frames]
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
            raise RuntimeError(f"DINO 输出 token 数不足：{tuple(tokens.shape)}")
        patch_grid = tokens[:, -patch_count:].reshape(-1, grid, grid, tokens.shape[-1])
        if grid % 4 != 0:
            raise ValueError(f"patch grid 必须能被4整除，当前为 {grid}×{grid}")

        block = grid // 4
        valid_grid = np.stack(
            [cv2.resize(mask, (grid, grid), interpolation=cv2.INTER_AREA) for mask in valid_masks],
            axis=0,
        ).astype(np.float32) / 255.0
        valid_grid = torch.from_numpy(valid_grid).to(patch_grid.device, patch_grid.dtype)

        patch_6d = patch_grid.reshape(-1, 4, block, 4, block, tokens.shape[-1])
        valid_regions = valid_grid.reshape(-1, 4, block, 4, block)
        valid_sum = valid_regions.sum(dim=(2, 4)).clamp_min(1e-6)
        valid_mean = (patch_6d * valid_regions[..., None]).sum(dim=(2, 4)) / valid_sum[..., None]

        region_valid_mask = (valid_sum > 1e-6).reshape(-1, 16)
        if structure_weight is None:
            regions = valid_mean.reshape(-1, 16, tokens.shape[-1])
            degenerate = torch.zeros_like(region_valid_mask)
        else:
            weight_grid = cv2.resize(structure_weight, (grid, grid), interpolation=cv2.INTER_AREA)
            weight_grid = torch.from_numpy(weight_grid.astype(np.float32)).to(
                patch_grid.device, patch_grid.dtype
            )
            structure_regions = weight_grid.reshape(4, block, 4, block)
            effective_regions = valid_regions * structure_regions[None]
            structure_sum = effective_regions.sum(dim=(2, 4))
            weighted_sum = (patch_6d * effective_regions[..., None]).sum(dim=(2, 4))
            degenerate = (structure_sum <= 1e-6).reshape(-1, 16)
            regions = torch.where(
                (structure_sum > 1e-6)[..., None],
                weighted_sum / structure_sum.clamp_min(1e-6)[..., None],
                valid_mean,
            ).reshape(-1, 16, tokens.shape[-1])

        global_features = (
            outputs.pooler_output
            if getattr(outputs, "pooler_output", None) is not None
            else tokens[:, 0]
        )
        return (
            global_features.float().cpu(),
            regions.float().cpu(),
            region_valid_mask.cpu(),
            degenerate.cpu(),
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
        raise RuntimeError(f"无法打开视频：{path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_indices(total, fps, sample_fps, max_frames)
    raw_frames, timestamps = [], []
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
        (i for i, frame in enumerate(raw_frames) if frame_quality_passes(frame)),
        0,
    )
    reference_gray = cv2.cvtColor(raw_frames[reference_index], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    registered, registration_masks = [], []
    ecc_success, ecc_failure = 0, 0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)

    for index, frame in enumerate(raw_frames):
        if index == reference_index:
            registered.append(frame)
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
            registered.append(aligned)
            registration_masks.append(aligned_mask)
            ecc_success += 1
        except cv2.error:
            registered.append(frame)
            registration_masks.append(np.ones(frame.shape[:2], dtype=np.uint8) * 255)
            ecc_failure += 1

    stats = {
        "fps": fps,
        "decoded_selected_frames": len(registered),
        "ecc_success": ecc_success,
        "ecc_failure": ecc_failure,
        "reference_frame_index": int(reference_index),
        "reference_quality_passed": bool(frame_quality_passes(raw_frames[reference_index])),
        "preprocessing_version": PREPROCESSING_VERSION,
        "structure_weight_type": "reference_frame_multiscale_frangi",
        "structure_weight_fixed_across_frames": True,
    }
    return registered, raw_frames, registration_masks, np.asarray(timestamps, dtype=np.float32), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v1.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--use-frangi-weight",
        "--use-vessel-weight",
        dest="use_frangi_weight",
        action="store_true",
        help="启用P3：使用固定参考帧的多尺度Frangi软权重。",
    )
    args = parser.parse_args()
    if not args.use_frangi_weight:
        raise ValueError("P3提取必须使用 --use-frangi-weight。")

    config = load_config(args.config)
    manifest = discover_videos(config)
    data_cfg = config["data"]
    save_manifest(manifest, data_cfg["manifest_path"])
    cache_dir = Path(data_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = cache_dir / "raw_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    encoder = DINOv3Encoder(
        config["backbone"]["model_name"],
        int(data_cfg.get("image_size", 256)),
        torch.device(args.device),
    )

    rows = []
    for row in tqdm(manifest.to_dict("records"), desc="extract_p3"):
        target = cache_dir / f"{int(row['video_id']):05d}.pt"
        raw_sidecar = raw_dir / f"{int(row['video_id']):05d}.npz"
        if target.exists() and raw_sidecar.exists() and not args.overwrite:
            rows.append({**row, "cache_path": str(target.resolve()), "raw_frames_path": str(raw_sidecar.resolve()), "status": "exists"})
            continue

        registered, raw_frames, registration_masks, timestamps, stats = read_registered_frames(
            row["path"],
            float(data_cfg.get("sample_fps", 2.0)),
            int(data_cfg.get("max_frames", 128)),
        )
        padded_frames, view_masks = [], []
        for frame, registration_mask in zip(registered, registration_masks):
            padded, view_mask = resize_with_padding(frame, encoder.image_size)
            registration_mask = cv2.resize(
                registration_mask,
                (encoder.image_size, encoder.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            padded_frames.append(padded)
            view_masks.append(cv2.bitwise_and(view_mask, registration_mask))

        reference_index = int(stats["reference_frame_index"])
        reference_weight = make_reference_frangi_weight(
            registered[reference_index],
            encoder.image_size,
            view_masks[reference_index],
        )

        global_parts, teacher_region_parts, region_parts = [], [], []
        region_valid_parts, degenerate_parts = [], []
        for start in range(0, len(padded_frames), args.batch_size):
            frames_batch = padded_frames[start:start + args.batch_size]
            masks_batch = view_masks[start:start + args.batch_size]
            teacher_global, teacher_region, valid_part, _ = encoder.encode(
                frames_batch, masks_batch, None,
            )
            global_part, region_part, _, degenerate_part = encoder.encode(
                frames_batch, masks_batch, reference_weight,
            )
            global_parts.append(teacher_global)
            teacher_region_parts.append(teacher_region)
            region_parts.append(region_part)
            region_valid_parts.append(valid_part)
            degenerate_parts.append(degenerate_part)

        np.savez_compressed(
            raw_sidecar,
            frames=np.stack(raw_frames),
            registration_masks=np.stack(registration_masks),
            view_masks=np.stack(view_masks),
            reference_frangi_weight=reference_weight,
        )
        payload = {
            "global_features": torch.cat(global_parts).half(),
            "teacher_region_features": torch.cat(teacher_region_parts).half(),
            "region_features": torch.cat(region_parts).half(),
            "region_valid_mask": torch.cat(region_valid_parts),
            "frangi_degenerate_mask": torch.cat(degenerate_parts),
            "timestamps": torch.from_numpy(timestamps),
            "video_id": int(row["video_id"]),
            "animal_id": str(row["animal_id"]),
            "label": int(row["label"]),
            "video_path": row["path"],
            "raw_frames_path": str(raw_sidecar.resolve()),
            "preprocessing_version": PREPROCESSING_VERSION,
            "structure_weight_type": "reference_frame_multiscale_frangi",
            "structure_weight_fixed_across_frames": True,
            "video_stats": stats,
        }
        torch.save(payload, target)
        rows.append({**row, "cache_path": str(target.resolve()), "raw_frames_path": str(raw_sidecar.resolve()), "status": "created", **stats})

    audit = Path(data_cfg["output_dir"]) / "feature_extraction_manifest_p3.csv"
    audit.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(audit, index=False, encoding="utf-8-sig")
    print(json.dumps({
        "videos": len(rows),
        "cache_dir": str(cache_dir.resolve()),
        "audit": str(audit.resolve()),
        "preprocessing_version": PREPROCESSING_VERSION,
        "structure_weight_type": "reference_frame_multiscale_frangi",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
