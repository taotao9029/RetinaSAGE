"""Align replacement-video P2 cache IDs with the canonical B manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", default="runs/B/feature_extraction_manifest.csv")
    parser.add_argument("--input-manifest", default="runs/P2_replacement/input_manifest.csv")
    parser.add_argument("--audit", default="runs/P2_replacement/feature_extraction_manifest.csv")
    parser.add_argument("--cache-dir", default="cache/dinov3_vits16_P2")
    args = parser.parse_args()

    canonical = pd.read_csv(args.canonical, encoding="gb18030")
    local = pd.read_csv(args.input_manifest, encoding="utf-8-sig")
    audit = pd.read_csv(args.audit, encoding="utf-8-sig")
    cache = Path(args.cache_dir).resolve()
    canonical_ids = dict(zip(canonical["animal_id"], canonical["video_id"]))
    assert len(canonical_ids) == len(canonical), "主清单 animal_id 不唯一"
    assert local["animal_id"].is_unique and audit["animal_id"].is_unique
    assert set(local["animal_id"]) == set(audit["animal_id"])
    assert set(local["animal_id"]).issubset(canonical_ids)
    assert (local["label"] == 1).all() and (audit["label"] == 1).all()

    moves = []
    for row in local.itertuples(index=False):
        old_id = int(row.video_id)
        new_id = int(canonical_ids[row.animal_id])
        old_pt = cache / f"{old_id:05d}.pt"
        new_pt = cache / f"{new_id:05d}.pt"
        old_npz = cache / "raw_frames" / f"{old_id:05d}.npz"
        new_npz = cache / "raw_frames" / f"{new_id:05d}.npz"
        if not old_pt.is_file() or not old_npz.is_file():
            raise FileNotFoundError(f"缺少原始缓存: {old_pt} 或 {old_npz}")
        if new_pt.exists() or new_npz.exists():
            raise FileExistsError(f"目标已存在，拒绝覆盖: {new_pt} 或 {new_npz}")
        payload = torch.load(old_pt, map_location="cpu", weights_only=False)
        if payload["animal_id"] != row.animal_id or int(payload["video_id"]) != old_id:
            raise ValueError(f"缓存元数据与清单不一致: {old_pt}")
        if Path(payload["video_path"]).name != Path(row.path).name:
            raise ValueError(f"视频来源与清单不一致: {old_pt}")
        moves.append((row.animal_id, old_pt, new_pt, old_npz, new_npz, payload, new_id))

    # Write all new PTs first; leave originals untouched until every output is verified.
    for animal_id, old_pt, new_pt, old_npz, new_npz, payload, new_id in moves:
        payload["video_id"] = new_id
        payload["raw_frames_path"] = str(new_npz)
        temp = new_pt.with_suffix(".pt.tmp")
        if temp.exists():
            raise FileExistsError(f"临时文件已存在: {temp}")
        torch.save(payload, temp)
        check = torch.load(temp, map_location="cpu", weights_only=False)
        if check["animal_id"] != animal_id or int(check["video_id"]) != new_id:
            raise ValueError(f"写入校验失败: {temp}")
        os.replace(temp, new_pt)

    for _, _, _, old_npz, new_npz, _, _ in moves:
        old_npz.rename(new_npz)

    for frame, path in ((local, Path(args.input_manifest)), (audit, Path(args.audit))):
        frame["video_id"] = frame["animal_id"].map(canonical_ids).astype(int)
        if "cache_path" in frame:
            frame["cache_path"] = frame["video_id"].map(lambda i: str(cache / f"{i:05d}.pt"))
        if "raw_frames_path" in frame:
            frame["raw_frames_path"] = frame["video_id"].map(
                lambda i: str(cache / "raw_frames" / f"{i:05d}.npz")
            )
        temp = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temp, index=False, encoding="utf-8-sig")
        os.replace(temp, path)

    for _, old_pt, new_pt, _, new_npz, _, _ in moves:
        if not new_pt.is_file() or not new_npz.is_file():
            raise RuntimeError(f"重命名后文件缺失: {new_pt}")
        old_pt.unlink()

    for animal_id, _, new_pt, _, _, _, new_id in moves:
        print(f"{animal_id}: {new_id:05d} -> {new_pt}")


if __name__ == "__main__":
    main()
