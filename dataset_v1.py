"""Dataset discovery and manifest utilities for OCTA-VIT V1."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalise_name(value: Any) -> str:
    return Path(str(value)).name.strip().lower()


def get_animal_id(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"(?:_aug|_augmentation)[-_]?\d+$", "", stem, flags=re.I)


def _find_column(columns, candidates, contains=()):
    clean = {str(c).strip(): c for c in columns}
    for candidate in candidates:
        if candidate in clean:
            return clean[candidate]
    for column in columns:
        text = str(column).strip().lower()
        if any(token.lower() in text for token in contains):
            return column
    return None


def parse_label(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("标签为空")
    text = str(value).strip().lower()
    if text in {"是", "yes", "y", "true", "1", "stroke", "positive"}:
        return 1
    if text in {"否", "no", "n", "false", "0", "healthy", "negative"}:
        return 0
    try:
        number = int(float(text))
    except ValueError as exc:
        raise ValueError(f"无法解析标签: {value!r}") from exc
    if number not in {0, 1}:
        raise ValueError(f"标签必须为0/1或是/否，得到: {value!r}")
    return number


def load_labels(label_xlsx: str | Path) -> dict[str, int]:
    path = Path(label_xlsx)
    alternatives = [
        path,
        Path(str(path).replace("（", "(").replace("）", ")")),
        Path(str(path).replace("(", "（").replace(")", "）")),
    ]
    selected = next((item for item in alternatives if item.exists()), None)
    if selected is None:
        raise FileNotFoundError(
            "找不到标签 Excel。尝试过: " + ", ".join(str(item) for item in alternatives)
        )

    frame = pd.read_excel(selected)
    name_col = _find_column(
        frame.columns,
        ["AVI文件名", "文件名", "filename", "file_name", "video", "video_name"],
        contains=("文件名", "filename", "video"),
    )
    label_col = _find_column(
        frame.columns,
        ["是否脑卒", "label", "标签", "class", "target"],
        contains=("脑卒", "label", "标签", "class", "target"),
    )
    if name_col is None or label_col is None:
        raise ValueError(
            f"标签表需要文件名和标签字段，当前字段为: {list(frame.columns)}"
        )

    result: dict[str, int] = {}
    for _, row in frame.iterrows():
        name = normalise_name(row[name_col])
        if not name or name == "nan":
            continue
        label = parse_label(row[label_col])
        if name in result and result[name] != label:
            raise ValueError(f"同一视频出现冲突标签: {name}")
        result[name] = label
        result.setdefault(Path(name).stem, label)
    return result


def discover_videos(config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    label_map = load_labels(data_cfg["label_xlsx"])
    extensions = {str(ext).lower() for ext in data_cfg.get("extensions", VIDEO_EXTENSIONS)}
    rows = []
    for root_value in data_cfg["video_roots"]:
        root = Path(root_value)
        if not root.is_dir():
            raise FileNotFoundError(f"视频目录不存在: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            name = normalise_name(path.name)
            stem = normalise_name(path.stem)
            label = label_map.get(name, label_map.get(stem))
            if label is None:
                raise ValueError(f"视频没有匹配标签: {path}")
            animal_id = get_animal_id(path.name)
            rows.append(
                {
                    "path": str(path.resolve()),
                    "filename": path.name,
                    "source_folder": root.name,
                    "animal_id": animal_id,
                    "label": int(label),
                }
            )

    if not rows:
        raise FileNotFoundError("指定目录中没有找到视频")
    result = pd.DataFrame(rows).sort_values(["animal_id", "filename"]).reset_index(drop=True)
    animal_labels = result.groupby("animal_id")["label"].nunique()
    if (animal_labels > 1).any():
        bad = animal_labels[animal_labels > 1].index.tolist()
        raise ValueError(f"同一 animal_id 出现多个标签: {bad[:10]}")
    duplicates = result["animal_id"].duplicated(keep=False)
    if duplicates.any():
        duplicate_ids = result.loc[duplicates, "animal_id"].unique().tolist()
        raise ValueError(
            "V1 默认每只动物对应一个原始视频，但发现重复 animal_id: "
            f"{duplicate_ids[:10]}。请先明确多视频如何聚合。"
        )
    result.insert(0, "video_id", range(len(result)))
    return result


def save_manifest(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")

