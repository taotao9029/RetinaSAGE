'''
说明：

1. 使用 RETFound OCT 版本 ViT-L/16，输出 1024 维全局特征
2. 输出 pt：`{"global_features": torch.Tensor(T,1024)}`，T 为视频帧数，不需要 region 特征（B8 只用 global）
3. 复用 `dataset_v1.discover_videos`，manifest 格式和`extract_features_v1.py`保持一致
4. 命令：`python extract_retfound_features.py --config config_v1_RETF.yaml --device cuda --batch-size 8`
'''
from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import yaml
from tqdm import tqdm
from dataset_v1 import discover_videos, load_config, save_manifest
import pandas as pd

# -------------------------- RETFound ViT-L backbone definition --------------------------
class RETFoundEncoder:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.image_size = 224
        # Load RETFound full checkpoint (encoder + decoder)
        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt["model"]

        # Only keep encoder weights, drop MAE decoder
        encoder_sd = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_sd[k] = v

        # Build ViT-L/16
        import timm
        self.model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
        )
        missing, unexpected = self.model.load_state_dict(encoder_sd, strict=False)
        print(f"RETFound load: missing keys {len(missing)}, unexpected {len(unexpected)}")
        self.model.eval()
        self.model.to(self.device)

        # Image normalize same as RETFound pretrain
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    @torch.no_grad()
    def extract_frame_feature(self, img: np.ndarray):
        """img: H,W,3 RGB uint8"""
        h,w = img.shape[:2]
        img = cv2.resize(img, (self.image_size, self.image_size))
        tensor = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        feat = self.model(tensor)
        return feat

    @torch.no_grad()
    def extract_video_feature(self, frames: list[np.ndarray]) -> torch.Tensor:
        """
        frames: list of RGB image array
        return: (T, 1024) global feature for each frame
        """
        feats = []
        for frame in frames:
            f = self.extract_frame_feature(frame)
            feats.append(f)
        feats = torch.cat(feats, dim=0)
        return feats


def read_video_frames(video_path: Path, sample_fps: float=2.0, max_frames:int=128):
    """Read video and sample frames at sample_fps"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / sample_fps))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            if len(frames) >= max_frames:
                break
        idx +=1
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Extract RETFound global features for B8")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
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

        # Skip already extracted
        if pt_path.exists():
            manifest_rows.append({
                "video_id": video_id,
                "animal_id": animal_id,
                "label": label,
                "cache_path": str(pt_path.resolve()),
                "video_path": str(video_path.resolve())
            })
            continue

        try:
            frames = read_video_frames(video_path, sample_fps=sample_fps, max_frames=max_frames)
            if len(frames) == 0:
                print(f"Warning: empty video, skip {video_path}")
                continue
            global_feat = encoder.extract_video_feature(frames)
            # B8 only stores global features, no 4x4 regional graph features
            save_dict = {
                "global_features": global_feat.cpu()
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
