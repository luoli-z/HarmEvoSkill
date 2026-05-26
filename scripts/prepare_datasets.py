# -*- coding: utf-8 -*-
"""Prepare supported meme datasets into the framework JSONL layout.

The script does not download restricted datasets. Place the original files under
``data/raw/<DATASET>/`` according to README.md, then run this script to create
``data/<DATASET>/train.jsonl``, ``val.jsonl`` when available, ``test.jsonl``,
and an ``images/`` directory.
"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SUPPORTED = ["FHM", "HarM", "Harm-C", "Harm-P", "MultiOFF", "PrideMM"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for experiments.")
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED + ["all"],
        default="all",
        help="Dataset to prepare.",
    )
    parser.add_argument("--raw_root", default="data/raw", help="Root with original datasets.")
    parser.add_argument("--out_root", default="data", help="Output data root.")
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images into data/<dataset>/images. By default images are left in place.",
    )
    args = parser.parse_args()

    datasets = SUPPORTED if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        prepare_dataset(
            dataset=dataset,
            raw_root=Path(args.raw_root),
            out_root=Path(args.out_root),
            copy_images=args.copy_images,
        )


def prepare_dataset(dataset: str, raw_root: Path, out_root: Path, copy_images: bool) -> None:
    canonical = "Harm-C" if dataset == "HarM" else dataset
    raw_dir = raw_root / canonical
    out_dir = out_root / canonical
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    if canonical == "FHM":
        prepare_jsonl_dataset(
            raw_dir=raw_dir,
            out_dir=out_dir,
            splits={"train": "train.jsonl", "test": "test.jsonl"},
            image_key="img",
            text_key="text",
            label_key="label",
        )
    elif canonical in {"Harm-C", "Harm-P"}:
        prepare_jsonl_dataset(
            raw_dir=raw_dir,
            out_dir=out_dir,
            splits={
                "train": first_existing(raw_dir, ["train.jsonl", "train_v1.jsonl"]),
                "val": first_existing(raw_dir, ["val.jsonl", "val_v1.jsonl"]),
                "test": first_existing(raw_dir, ["test.jsonl", "test_v1.jsonl"]),
            },
            image_key="image",
            text_key="text",
            label_key="labels",
        )
    elif canonical == "MultiOFF":
        prepare_multioff(raw_dir, out_dir)
    elif canonical == "PrideMM":
        prepare_pridemm(raw_dir, out_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if copy_images:
        copy_images_if_present(raw_dir / "images", out_dir / "images")

    print(f"Prepared {dataset} -> {out_dir}")


def prepare_jsonl_dataset(
    raw_dir: Path,
    out_dir: Path,
    splits: Dict[str, Optional[str]],
    image_key: str,
    text_key: str,
    label_key: str,
) -> None:
    for split, filename in splits.items():
        if not filename:
            continue
        src = raw_dir / filename
        if not src.exists():
            if split in {"val"}:
                continue
            raise FileNotFoundError(f"Missing {split} file: {src}")
        rows = []
        for idx, item in enumerate(read_jsonl(src)):
            rows.append({
                "id": item.get("id", item.get("idx", idx)),
                image_key: item.get(image_key),
                text_key: item.get(text_key, ""),
                label_key: item.get(label_key),
            })
        write_jsonl(out_dir / f"{split}.jsonl", rows)


def prepare_multioff(raw_dir: Path, out_dir: Path) -> None:
    files = {
        "train": "Training_meme_dataset.csv",
        "val": "Validation_meme_dataset.csv",
        "test": "Testing_meme_dataset.csv",
    }
    for split, filename in files.items():
        src = raw_dir / filename
        if not src.exists():
            if split == "val":
                continue
            raise FileNotFoundError(f"Missing MultiOFF {split} file: {src}")
        rows = []
        for idx, row in enumerate(read_csv(src)):
            rows.append({
                "id": f"{split}_{idx}",
                "image_name": row.get("image_name") or row.get("image") or row.get("name"),
                "sentence": row.get("sentence") or row.get("text") or "",
                "label": normalize_multioff_label(row.get("label")),
            })
        write_jsonl(out_dir / f"{split}.jsonl", rows)


def prepare_pridemm(raw_dir: Path, out_dir: Path) -> None:
    src = raw_dir / "PrideMM.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing PrideMM.csv: {src}")
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for idx, row in enumerate(read_csv(src)):
        split = str(row.get("split", "train") or "train").strip().lower()
        if split in {"dev", "valid", "validation"}:
            split = "val"
        if split not in by_split:
            split = "train"
        by_split[split].append({
            "id": row.get("id", idx),
            "name": row.get("name") or row.get("image") or row.get("image_name"),
            "text": row.get("text", ""),
            "hate": int(str(row.get("hate", "0")).strip() or 0),
            "target": row.get("target"),
            "stance": row.get("stance"),
            "humour": row.get("humour"),
        })
    for split, rows in by_split.items():
        if rows:
            write_jsonl(out_dir / f"{split}.jsonl", rows)


def first_existing(raw_dir: Path, names: List[str]) -> Optional[str]:
    for name in names:
        if (raw_dir / name).exists():
            return name
    return names[0] if names else None


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def read_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def normalize_multioff_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"non-offensiv", "non-offensive", "not offensive", "0"}:
        return "non-offensive"
    return "offensive"


def copy_images_if_present(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"Image directory not found, skipping copy: {src}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    for file_path in src.iterdir():
        if file_path.is_file():
            shutil.copy2(file_path, dst / file_path.name)


if __name__ == "__main__":
    main()
