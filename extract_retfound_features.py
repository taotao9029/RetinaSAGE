'''Extract the frozen RETFound features required by task-book baseline B8.

B8 is defined as RETFound-MAE + frame-global features + video mean pooling +
logistic regression. The cache keeps both the per-frame ``global_features``
(for compatibility with existing B8/B9 loaders) and the explicit
``mean_feature`` used by the baseline definition.
python extract_retfound_features.py \
    --config config_v1_RETF.yaml \
    --device cuda \
    --batch-size 8 \
    --overwrite
'''
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from dataset_v1 import discover_videos, load_config, save_manifest
import pandas as pd


RETFOUND_FEATURE_VERSION = "b8_retfound_frame_global_mean_pool_v2"

# -------------------------- RETFound ViT-L backbone definition --------------------------
class RETFoundEncoder:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.image_size = 224
        # The local RETFound checkpoint stores ViT keys directly (without an
        # ``encoder.`` prefix) and also contains MAE decoder weights.
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:  # torch < 2.6
            ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

        import timm
        self.model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
        )
        model_state = self.model.state_dict()
        encoder_sd = {}
        for key, value in state_dict.items():
            key = key.removeprefix("module.").removeprefix("encoder.")
            if key in model_state and model_state[key].shape == value.shape:
                encoder_sd[key] = value
        missing, unexpected = self.model.load_state_dict(encoder_sd, strict=False)
        if not encoder_sd:
            raise RuntimeError(f"No ViT encoder weights found in RETFound checkpoint: {model_path}")
        print(
            f"RETFound load: transferred {len(encoder_sd)}/{len(model_state)} tensors, "
            f"missing {len(missing)}, unexpected {len(unexpected)}"
        )
        self.model.eval()
        self.model.to(self.device)

        # Image normalize same as RETFound pretrain
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    @torch.no_grad()
    def _preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        tensors = []
        for frame in frames:
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Expected RGB HxWx3 frame, got {frame.shape}")
            resized = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            tensors.append(torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0)
        batch = torch.stack(tensors).to(self.device)
        return (batch - self.mean) / self.std

    @torch.no_grad()
    def extract_video_feature(self, frames: list[np.ndarray], batch_size: int = 8):
        """
        frames: list of RGB image array
        return: (T, 1024) global feature for each frame
        """
        feats = []
        for start in range(0, len(frames), max(1, int(batch_size))):
            with torch.inference_mode():
                batch = self._preprocess(frames[start:start + batch_size])
                feats.append(self.model(batch).float().cpu())
        frame_features = torch.cat(feats, dim=0)
        return frame_features, frame_features.mean(dim=0)


def read_video_frames(video_path: Path, sample_fps: float = 2.0, max_frames: int = 128):
    """Read video and sample frames at sample_fps"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    interval = max(1, int(round(fps / max(sample_fps, 1e-6))))

    frames = []
    timestamps = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            timestamps.append(idx / fps)
            if len(frames) >= max_frames:
                break
        idx +=1
    cap.release()
    return frames, np.asarray(timestamps, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract RETFound global features for B8")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="重新提取已有缓存。")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]

    cache_dir = Path(data_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(data_cfg["manifest_path"])

    sample_fps = data_cfg["sample_fps"]
    max_frames = data_cfg["max_frames"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = config["backbone"]["model_name"]
    encoder = RETFoundEncoder(ckpt_path, device)

    # discover_videos 返回DataFrame
    video_df = discover_videos(config)

    manifest_rows = []
    for _, vid_item in tqdm(video_df.iterrows(), total=len(video_df)):
        video_path = Path(vid_item["path"])
        animal_id = vid_item["animal_id"]
        label = int(vid_item["label"])
        video_id = int(vid_item["video_id"])

        pt_path = cache_dir / f"{video_id:05d}.pt"

        compatible_cache = False
        if pt_path.exists() and not args.overwrite:
            try:
                cache = torch.load(pt_path, map_location="cpu", weights_only=False)
            except TypeError:
                cache = torch.load(pt_path, map_location="cpu")
            compatible_cache = all(
                key in cache for key in ("global_features", "mean_feature", "timestamps", "valid_mask")
            )
        if compatible_cache:
            manifest_rows.append({
                "video_id": video_id,
                "animal_id": animal_id,
                "label": label,
                "cache_path": str(pt_path.resolve()),
                "video_path": str(video_path.resolve())
            })
            continue

        try:
            frames, timestamps = read_video_frames(video_path, sample_fps=sample_fps, max_frames=max_frames)
            if len(frames) == 0:
                print(f"Warning: empty video, skip {video_path}")
                continue
            global_feat, mean_feat = encoder.extract_video_feature(frames, batch_size=args.batch_size)
            save_dict = {
                "global_features": global_feat.cpu(),
                "mean_feature": mean_feat.cpu(),
                "timestamps": torch.from_numpy(timestamps),
                "valid_mask": torch.ones(len(frames), dtype=torch.bool),
                "feature_spec": {
                    "version": RETFOUND_FEATURE_VERSION,
                    "backbone": "RETFound ViT-L/16",
                    "feature_dim": int(global_feat.shape[-1]),
                    "frame_pooling": "arithmetic_mean_over_valid_sampled_frames",
                    "input_size": int(encoder.image_size),
                    "sample_fps": float(sample_fps),
                },
            }
            torch.save(save_dict, pt_path)
            manifest_rows.append({
                "video_id": video_id,
                "animal_id": animal_id,
                "label": label,
                "cache_path": str(pt_path.resolve()),
                "video_path": str(video_path.resolve())
            })
        except Exception as e:
            print(f"ERROR processing {video_path}: {str(e)}")

    save_manifest(pd.DataFrame(manifest_rows), manifest_path)
    print(f"Done! B8 manifest saved at {manifest_path}, total valid videos: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
