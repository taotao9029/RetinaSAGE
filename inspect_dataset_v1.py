"""Audit raw-video availability before feature extraction."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dataset_v1 import discover_videos, load_config, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    frame = discover_videos(config)
    durations, fps_values, frames = [], [], []
    for path in frame["path"]:
        capture = cv2.VideoCapture(path)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        fps = fps if np.isfinite(fps) and fps > 0 else np.nan
        fps_values.append(fps)
        frames.append(count)
        durations.append(count / fps if np.isfinite(fps) else np.nan)
    frame["fps"] = fps_values
    frame["frame_count"] = frames
    frame["duration_s"] = durations
    save_manifest(frame, config["data"]["manifest_path"])
    print(frame.groupby("label").size().rename("videos").to_string())
    print("total_videos", len(frame))
    print("total_animals", frame["animal_id"].nunique())
    print("duration_s", frame["duration_s"].describe().to_string())
    print("fps", frame["fps"].describe().to_string())
    print("manifest", Path(config["data"]["manifest_path"]).resolve())


if __name__ == "__main__":
    main()

