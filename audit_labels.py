from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class LabelIssue:
    split: str
    image: str
    label_file: str
    issue_type: str
    details: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit YOLO label quality and optionally apply safe fixes.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/hardhat_yolo",
        help="Path to the YOLO dataset root.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply safe fixes such as clamping normalized boxes and dropping malformed rows.",
    )
    parser.add_argument(
        "--tiny-box-threshold",
        type=float,
        default=0.0015,
        help="Boxes smaller than this normalized area ratio are flagged as suspicious.",
    )
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=0.95,
        help="IoU threshold for flagging near-duplicate boxes of the same class.",
    )
    return parser.parse_args()


def get_image_map(images_dir: Path) -> dict[str, Path]:
    image_map: dict[str, Path] = {}
    if not images_dir.exists():
        return image_map
    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_map[image_path.stem] = image_path
    return image_map


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float) -> tuple[float, float, float, float]:
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return x1, y1, x2, y2


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-9, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def audit_split(
    dataset_root: Path,
    split: str,
    apply_fixes: bool,
    tiny_box_threshold: float,
    duplicate_iou_threshold: float,
    valid_class_ids: set[int],
) -> tuple[list[LabelIssue], Counter]:
    issues: list[LabelIssue] = []
    counters: Counter = Counter()
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    image_map = get_image_map(images_dir)
    label_map = {path.stem: path for path in labels_dir.glob("*.txt")} if labels_dir.exists() else {}

    for stem, image_path in image_map.items():
        if stem not in label_map:
            issues.append(LabelIssue(split, image_path.name, "", "missing_label_file", "Image has no matching label file"))
            counters["missing_label_file"] += 1

    for stem, label_path in label_map.items():
        if stem not in image_map:
            issues.append(LabelIssue(split, "", label_path.name, "missing_image_file", "Label file has no matching image"))
            counters["missing_image_file"] += 1
            continue

        image_path = image_map[stem]
        image = cv2.imread(str(image_path))
        if image is None:
            issues.append(LabelIssue(split, image_path.name, label_path.name, "corrupt_image", "Image could not be read"))
            counters["corrupt_image"] += 1
            continue

        height, width = image.shape[:2]
        seen_boxes_by_class: defaultdict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
        updated_lines: list[str] = []

        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 5:
                issues.append(
                    LabelIssue(split, image_path.name, label_path.name, "malformed_row", f"Line {line_number} does not have 5 fields"))
                counters["malformed_row"] += 1
                continue

            try:
                class_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                issues.append(
                    LabelIssue(split, image_path.name, label_path.name, "non_numeric_row", f"Line {line_number} contains non-numeric values"))
                counters["non_numeric_row"] += 1
                continue

            if class_id not in valid_class_ids:
                issues.append(
                    LabelIssue(split, image_path.name, label_path.name, "invalid_class_id", f"Line {line_number} has class {class_id}"))
                counters["invalid_class_id"] += 1
                continue

            original = (xc, yc, w, h)
            xc, yc, w, h = clamp01(xc), clamp01(yc), clamp01(w), clamp01(h)
            if w <= 0 or h <= 0:
                issues.append(
                    LabelIssue(split, image_path.name, label_path.name, "non_positive_box", f"Line {line_number} has non-positive width/height"))
                counters["non_positive_box"] += 1
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h)
            if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                issues.append(
                    LabelIssue(split, image_path.name, label_path.name, "box_out_of_bounds", f"Line {line_number} exceeded image bounds"))
                counters["box_out_of_bounds"] += 1
                x1, y1, x2, y2 = clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue

            area_ratio = w * h
            if area_ratio < tiny_box_threshold:
                pixel_area = area_ratio * width * height
                issues.append(
                    LabelIssue(split, image_path.name, label_path.name, "tiny_box", f"Line {line_number} tiny box area {pixel_area:.2f}px"))
                counters["tiny_box"] += 1

            box_xyxy = (x1, y1, x2, y2)
            for previous in seen_boxes_by_class[class_id]:
                if iou(previous, box_xyxy) >= duplicate_iou_threshold:
                    issues.append(
                        LabelIssue(split, image_path.name, label_path.name, "duplicate_box", f"Line {line_number} overlaps same-class box with IoU >= {duplicate_iou_threshold}"))
                    counters["duplicate_box"] += 1
                    break
            seen_boxes_by_class[class_id].append(box_xyxy)

            if original != (xc, yc, w, h):
                counters["safe_fixes_available"] += 1

            updated_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if apply_fixes:
            label_path.write_text("\n".join(updated_lines), encoding="utf-8")

    return issues, counters


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    valid_class_ids = {0, 1}
    all_issues: list[LabelIssue] = []
    totals: Counter = Counter()

    for split in ["train", "val", "test"]:
        split_issues, split_counts = audit_split(
            dataset_root=dataset_root,
            split=split,
            apply_fixes=args.apply,
            tiny_box_threshold=args.tiny_box_threshold,
            duplicate_iou_threshold=args.duplicate_iou_threshold,
            valid_class_ids=valid_class_ids,
        )
        all_issues.extend(split_issues)
        totals.update(split_counts)

    report_dir = dataset_root / "audit_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "label_quality_report.csv"
    summary_path = report_dir / "label_quality_summary.txt"

    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "image", "label_file", "issue_type", "details"])
        for issue in all_issues:
            writer.writerow([issue.split, issue.image, issue.label_file, issue.issue_type, issue.details])

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Label Quality Audit Summary\n")
        handle.write(f"Dataset: {dataset_root.resolve()}\n")
        handle.write(f"Apply fixes: {args.apply}\n\n")
        for key in sorted(totals):
            handle.write(f"{key}: {totals[key]}\n")
        handle.write(f"\nTotal issues logged: {len(all_issues)}\n")
        handle.write(f"Detailed report: {report_path}\n")

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
