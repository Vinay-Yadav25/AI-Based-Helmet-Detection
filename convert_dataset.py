from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


CLASS_MAP = {
    "helmet": 0,
    "head": 1,
}

CLASS_NAMES = {
    0: "helmet",
    1: "no_helmet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Hardhat Pascal VOC dataset to YOLO format."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="dataset/Hardhat",
        help="Path to the Pascal VOC style Hardhat dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/hardhat_yolo",
        help="Output path for the converted YOLO dataset.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio sampled from the Train set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_box(size: tuple[int, int], box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    width, height = size
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def parse_annotation(xml_path: Path) -> tuple[str, list[str], Counter]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size_node = root.find("size")
    if filename is None or size_node is None:
        raise ValueError(f"Invalid annotation file: {xml_path}")

    width = int(float(size_node.findtext("width", "0")))
    height = int(float(size_node.findtext("height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in: {xml_path}")

    yolo_lines: list[str] = []
    stats: Counter = Counter()

    for obj in root.findall("object"):
        raw_label = (obj.findtext("name") or "").strip().lower()
        if raw_label not in CLASS_MAP:
            stats[f"ignored:{raw_label or 'unknown'}"] += 1
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))

        xmin = min(max(xmin, 0.0), width)
        ymin = min(max(ymin, 0.0), height)
        xmax = min(max(xmax, 0.0), width)
        ymax = min(max(ymax, 0.0), height)

        if xmax <= xmin or ymax <= ymin:
            continue

        class_id = CLASS_MAP[raw_label]
        x_center, y_center, box_width, box_height = convert_box(
            (width, height),
            (xmin, ymin, xmax, ymax),
        )
        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
        stats[CLASS_NAMES[class_id]] += 1

    return filename, yolo_lines, stats


def copy_sample(image_src: Path, label_lines: list[str], output_root: Path, split: str) -> None:
    image_dst = output_root / "images" / split / image_src.name
    label_dst = output_root / "labels" / split / f"{image_src.stem}.txt"
    shutil.copy2(image_src, image_dst)
    label_dst.write_text("\n".join(label_lines), encoding="utf-8")


def convert_split(source_root: Path, output_root: Path, split_name: str) -> tuple[list[tuple[Path, list[str]]], Counter]:
    image_dir = source_root / split_name / "JPEGImage"
    ann_dir = source_root / split_name / "Annotation"
    samples: list[tuple[Path, list[str]]] = []
    stats: Counter = Counter()

    for xml_path in sorted(ann_dir.glob("*.xml")):
        filename, yolo_lines, annotation_stats = parse_annotation(xml_path)
        stats.update(annotation_stats)
        image_path = image_dir / filename
        if not image_path.exists():
            continue
        if not yolo_lines:
            continue
        samples.append((image_path, yolo_lines))

    return samples, stats


def write_data_yaml(output_root: Path) -> None:
    yaml_content = "\n".join(
        [
            f"path: {output_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            "  0: helmet",
            "  1: no_helmet",
            "",
        ]
    )
    (output_root / "data.yaml").write_text(yaml_content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source)
    output_root = Path(args.output)
    ensure_dirs(output_root)

    train_samples, train_stats = convert_split(source_root, output_root, "Train")
    test_samples, test_stats = convert_split(source_root, output_root, "Test")

    rng = random.Random(args.seed)
    rng.shuffle(train_samples)
    val_count = max(1, int(len(train_samples) * args.val_ratio))
    val_samples = train_samples[:val_count]
    actual_train_samples = train_samples[val_count:]

    for image_path, label_lines in actual_train_samples:
        copy_sample(image_path, label_lines, output_root, "train")
    for image_path, label_lines in val_samples:
        copy_sample(image_path, label_lines, output_root, "val")
    for image_path, label_lines in test_samples:
        copy_sample(image_path, label_lines, output_root, "test")

    write_data_yaml(output_root)

    print("Conversion complete.")
    print(f"Output dataset: {output_root.resolve()}")
    print(f"Train samples: {len(actual_train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print("Train annotation stats:", dict(train_stats))
    print("Test annotation stats:", dict(test_stats))


if __name__ == "__main__":
    main()
