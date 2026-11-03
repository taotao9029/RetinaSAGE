"""Run OCTA_Fusion/Video2Events.py on the labeled replacement AVI batch.

All files in the source directory are stroke videos (label=1). The optional
unlimited mode disables only the original per-video event-count cap.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import math
import os
from pathlib import Path
import time


DEFAULT_CONVERTER = Path(
    "/home/wxy/Classification/syy/syy/strock_test/OCTA_pytorch/OCTA_Fusion/Video2Events.py"
)
DEFAULT_SOURCE = Path("/home/datasets/cv/TCT/脑卒/OCTAvideo/上传替换")
DEFAULT_OUTPUT = Path("/home/datasets/cv/TCT/脑卒/OCTAvideo/替换evnet")


def load_converter(path: Path):
    spec = importlib.util.spec_from_file_location("octa_video2events", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_audit(path: Path, rows: list[dict]) -> None:
    fields = ["source_path", "output_path", "label", "status", "fps",
              "decoded_frames", "sampled_frames", "event_count",
              "positive_events", "negative_events", "sample_limit_reached",
              "event_limit_reached", "event_limit_enabled", "device",
              "sample_fps", "elapsed_seconds", "error"]
    temp = path.with_name(path.name + ".partial")
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields}
                          for row in rows])
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--unlimited", action="store_true",
                        help="Disable the MAX_EVENTS cap (all sampled frames are retained)")
    parser.add_argument("--replace-existing", action="store_true",
                        help="Atomically replace existing output CSV after successful conversion")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--only", default=None, help="Process only one AVI filename")
    args = parser.parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(args.source)
    if not args.output.is_dir():
        raise FileNotFoundError(args.output)

    videos = sorted(path for path in args.source.iterdir()
                    if path.is_file() and path.suffix.lower() == ".avi")
    if args.only is not None:
        videos = [video for video in videos if video.name == args.only]
    if not videos:
        raise RuntimeError(f"No AVI videos found in {args.source}")
    stems = [video.stem.lower() for video in videos]
    if len(stems) != len(set(stems)):
        raise RuntimeError("Duplicate AVI stems would overwrite an event CSV")

    converter = load_converter(args.converter)
    if args.unlimited:
        converter.MAX_EVENTS = math.inf
    device = converter.DEVICE if args.device is None else converter.torch.device(args.device)
    if device.type == "cpu":
        converter.torch.set_num_threads(8)
    print(f"converter={args.converter}", flush=True)
    print(f"source={args.source} output={args.output}", flush=True)
    print(f"videos={len(videos)} label=1 device={device}", flush=True)
    print(f"sample_fps={converter.SAMPLE_FPS} max_sample_frames={converter.MAX_SAMPLE_FRAMES} max_events={'unlimited' if args.unlimited else converter.MAX_EVENTS}", flush=True)
    rows = []
    audit_path = args.output / "conversion_manifest.csv"
    failures = 0
    for index, video in enumerate(videos, start=1):
        dest = args.output / f"{video.stem}.csv"
        temp = args.output / f"{video.stem}.csv.partial"
        if dest.exists() and not args.replace_existing:
            print(f"[{index}/{len(videos)}] EXISTS {dest}", flush=True)
            rows.append({"source_path": str(video), "output_path": str(dest),
                         "label": 1, "status": "exists_skipped"})
            write_audit(audit_path, rows)
            continue
        if temp.exists():
            raise RuntimeError(f"Partial output exists; inspect before continuing: {temp}")

        print(f"[{index}/{len(videos)}] PROCESS {video.name}", flush=True)
        started = time.perf_counter()
        try:
            events, stats = converter.video2events_fast(
                video, output_path=temp, device=device, return_stats=True)
            del events
            gc.collect()
            if stats["status"] != "ok":
                raise RuntimeError(f"Conversion returned status={stats['status']}")
            if args.unlimited and (stats["event_limit_reached"] or stats["sample_limit_reached"]):
                raise RuntimeError("Conversion reached an event or sampled-frame limit")
            os.replace(temp, dest)
            row = {**stats, "source_path": str(video),
                   "output_path": str(dest), "label": 1,
                   "event_limit_enabled": not args.unlimited,
                   "device": str(device), "sample_fps": converter.SAMPLE_FPS,
                   "elapsed_seconds": round(time.perf_counter() - started, 2)}
            rows.append(row)
            print(f"[{index}/{len(videos)}] {stats['status']} events={stats['event_count']} sampled_frames={stats['sampled_frames']} decoded_frames={stats['decoded_frames']} elapsed={row['elapsed_seconds']}s", flush=True)
        except Exception as exc:
            failures += 1
            rows.append({"source_path": str(video), "output_path": str(dest),
                         "label": 1, "status": "failed", "error": repr(exc)})
            print(f"[{index}/{len(videos)}] FAILED {video.name}: {exc}", flush=True)
        write_audit(audit_path, rows)
    print(f"audit={audit_path} completed={len(videos)-failures} failed={failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
