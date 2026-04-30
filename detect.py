from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from utils.detector import HelmetDetector
from utils.video import process_image, process_video_file, process_video_stream


APP_DIR = Path(__file__).resolve().parent


def get_default_model_path() -> Path:
    candidates = sorted(
        APP_DIR.rglob("best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return APP_DIR / "runs" / "detect" / "models" / "helmet_detector_stable" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run helmet detection on an image, video, or webcam."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(get_default_model_path()),
        help="Path to the trained YOLO model.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: image path, video path, or webcam index.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the processed output for image, video, or webcam sources.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory for processed outputs.",
    )
    return parser.parse_args()


def is_webcam_source(source: str) -> bool:
    return source.isdigit()


def main() -> None:
    args = parse_args()
    detector = HelmetDetector(model_path=args.model, conf_threshold=args.conf)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_webcam_source(args.source):
        process_video_stream(
            detector=detector,
            source=int(args.source),
            save_output=args.save,
            output_path=output_dir / "webcam_detected.mp4" if args.save else None,
            window_name="Helmet Detection Webcam",
        )
        return

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise RuntimeError(f"Unable to read image: {source_path}")

        processed_frame, summary = process_image(detector, frame)
        cv2.imshow("Helmet Detection Image", processed_frame)
        print(summary.to_dict())
        if args.save:
            destination = output_dir / f"{source_path.stem}_detected{source_path.suffix}"
            cv2.imwrite(str(destination), processed_frame)
            print(f"Saved output to: {destination}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    if source_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        destination = output_dir / f"{source_path.stem}_detected.mp4"
        processed_path, summary = process_video_file(
            detector=detector,
            input_path=source_path,
            output_path=destination,
        )
        print(summary.to_dict())
        print(f"Processed video saved to: {processed_path}")
        return

    raise ValueError("Unsupported source. Use an image path, video path, or webcam index such as --source 0.")


if __name__ == "__main__":
    main()
