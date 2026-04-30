from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
import shutil


CLASS_MAP = {
    "hardhat": 0,
    "no-hardhat": 1,
}

CLASS_NAMES = {
    0: "helmet",
    1: "no_helmet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the external keremberke hard-hat dataset for YOLOv8 training."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="dataset/external/keremberke_hard_hat_detection",
        help="Path to the extracted external dataset root.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/external/keremberke_yolo",
        help="Output path for the YOLO-format dataset.",
    )
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def ensure_train_annotations(source_root: Path) -> Path:
    train_root = source_root / "train_extracted"
    annotation_path = train_root / "_annotations.coco.json"
    if annotation_path.exists():
        return annotation_path

    archive_path = source_root / "data" / "train.zip"
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing train archive: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        archive.extract("_annotations.coco.json", path=train_root)
    return annotation_path


def coco_bbox_to_yolo(
    bbox: list[float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x_min, y_min, box_width, box_height = bbox
    if width <= 0 or height <= 0 or box_width <= 0 or box_height <= 0:
        return None

    x_center = (x_min + box_width / 2.0) / width
    y_center = (y_min + box_height / 2.0) / height
    norm_width = box_width / width
    norm_height = box_height / height

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    norm_width = min(max(norm_width, 0.0), 1.0)
    norm_height = min(max(norm_height, 0.0), 1.0)

    if norm_width <= 0.0 or norm_height <= 0.0:
        return None
    return x_center, y_center, norm_width, norm_height


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def convert_split(
    image_root: Path,
    annotation_path: Path,
    output_root: Path,
    split_name: str,
) -> Counter:
    data = load_coco(annotation_path)
    categories = {item["id"]: item["name"].strip().lower() for item in data["categories"]}
    images = {item["id"]: item for item in data["images"]}
    annotations_by_image: dict[int, list[dict]] = {}
    stats: Counter = Counter()

    for annotation in data["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    for image_id, image_info in images.items():
        file_name = image_info["file_name"]
        src_image = image_root / file_name
        if not src_image.exists():
            stats["missing_image"] += 1
            continue

        label_lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            raw_name = categories.get(annotation["category_id"], "").strip().lower()
            if raw_name not in CLASS_MAP:
                stats[f"ignored:{raw_name or 'unknown'}"] += 1
                continue

            converted = coco_bbox_to_yolo(
                annotation["bbox"],
                int(image_info["width"]),
                int(image_info["height"]),
            )
            if converted is None:
                stats["invalid_bbox"] += 1
                continue

            class_id = CLASS_MAP[raw_name]
            x_center, y_center, box_width, box_height = converted
            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )
            stats[CLASS_NAMES[class_id]] += 1

        if not label_lines:
            continue

        dst_image = output_root / "images" / split_name / src_image.name
        dst_label = output_root / "labels" / split_name / f"{src_image.stem}.txt"
        shutil.copy2(src_image, dst_image)
        dst_label.write_text("\n".join(label_lines), encoding="utf-8")
        stats["images"] += 1

    return stats


def write_data_yaml(output_root: Path) -> None:
    yaml_content = "\n".join(
        [
            f"path: {output_root.resolve().as_posix()}",
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

    train_ann = ensure_train_annotations(source_root)
    val_ann = source_root / "valid_extracted" / "_annotations.coco.json"
    test_ann = source_root / "test_extracted" / "_annotations.coco.json"

    train_stats = convert_split(
        image_root=source_root / "train_extracted",
        annotation_path=train_ann,
        output_root=output_root,
        split_name="train",
    )
    val_stats = convert_split(
        image_root=source_root / "valid_extracted",
        annotation_path=val_ann,
        output_root=output_root,
        split_name="val",
    )
    test_stats = convert_split(
        image_root=source_root / "test_extracted",
        annotation_path=test_ann,
        output_root=output_root,
        split_name="test",
    )
    write_data_yaml(output_root)

    print("External dataset preparation complete.")
    print(f"Output dataset: {output_root.resolve()}")
    print("Train stats:", dict(train_stats))
    print("Val stats:", dict(val_stats))
    print("Test stats:", dict(test_stats))


if __name__ == "__main__":
    main()
