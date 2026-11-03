"""Restore canonical metadata in replacement P1 features without changing tensors."""

from __future__ import annotations

import os
from pathlib import Path

import torch


P1 = Path("cache/dinov3_vits16_P1")
BACKUP = Path("cache/dinov3_vits16_P1_backup_before_P2_20261028")
P2 = Path("cache/dinov3_vits16_P2")
FIELDS = ("video_id", "animal_id", "label", "video_path")
TENSORS = ("global_features", "region_features", "region_valid_mask", "timestamps")


def main() -> None:
    changes = []
    for video_id in range(21, 29):
        filename = f"{video_id:05d}.pt"
        target, backup, source = (folder / filename for folder in (P1, BACKUP, P2))
        for path in (target, backup, source):
            if not path.is_file():
                raise FileNotFoundError(path)
        current = torch.load(target, map_location="cpu", weights_only=False)
        original = torch.load(backup, map_location="cpu", weights_only=False)
        replacement = torch.load(source, map_location="cpu", weights_only=False)
        if set(current) != set(replacement):
            raise ValueError(f"P1/P2 字段不一致: {filename}")
        if any(not torch.equal(current[key], replacement[key]) for key in TENSORS):
            raise ValueError(f"P1/P2 特征不一致: {filename}")
        if int(original["video_id"]) != video_id or int(original["label"]) != 1:
            raise ValueError(f"备份元数据异常: {filename}")
        temp = target.with_suffix(".pt.tmp")
        if temp.exists():
            raise FileExistsError(temp)
        changes.append((target, temp, current, original, replacement))

    for target, temp, current, original, replacement in changes:
        for key in FIELDS:
            current[key] = original[key]
        torch.save(current, temp)
        checked = torch.load(temp, map_location="cpu", weights_only=False)
        if any(checked[key] != original[key] for key in FIELDS):
            raise RuntimeError(f"元数据校验失败: {temp}")
        if any(not torch.equal(checked[key], replacement[key]) for key in TENSORS):
            raise RuntimeError(f"特征校验失败: {temp}")
        if checked["raw_frames_path"] != replacement["raw_frames_path"]:
            raise RuntimeError(f"raw_frames_path 意外改变: {temp}")
        os.replace(temp, target)
        print(f"updated {target.name}: {checked['animal_id']}")


if __name__ == "__main__":
    main()
