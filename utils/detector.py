from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


CLASS_COLORS = {
    "helmet": (0, 200, 0),
    "no_helmet": (0, 0, 255),
}

DISPLAY_LABELS = {
    "helmet": "Helmet",
    "no_helmet": "No Helmet",
}

LABEL_ALIASES = {
    "helmet": "helmet",
    "hardhat": "helmet",
    "no_helmet": "no_helmet",
    "no-helmet": "no_helmet",
    "without_helmet": "no_helmet",
    "withouthelmet": "no_helmet",
    "head": "no_helmet",
}


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


class HelmetDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.4,
        min_box_area_ratio: float = 0.0015,
    ) -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_file}. Train the model first or update the path."
            )

        self.model = YOLO(str(model_file))
        self.conf_threshold = conf_threshold
        self.min_box_area_ratio = min_box_area_ratio
        self.class_names = self.model.names

    @staticmethod
    def normalize_label(label: str) -> str:
        return LABEL_ALIASES.get(label.strip().lower(), label.strip().lower())

    def predict(self, frame: Any) -> list[Detection]:
        frame_height, frame_width = frame.shape[:2]
        min_area = frame_height * frame_width * self.min_box_area_ratio
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            verbose=False,
            imgsz=640,
            classes=[0, 1],
        )
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                raw_label = str(self.class_names[cls_id])
                label = self.normalize_label(raw_label)
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                if (x2 - x1) * (y2 - y1) < min_area:
                    continue
                detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        xyxy=(x1, y1, x2, y2),
                    )
                )
        return detections

    @staticmethod
    def draw_detections(frame, detections: list[Detection]):
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy
            color = CLASS_COLORS.get(detection.label, (255, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            display_label = DISPLAY_LABELS.get(detection.label, detection.label.replace("_", " ").title())
            caption = f"{display_label} {detection.confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                caption,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2,
            )
            label_top = max(y1 - text_height - baseline - 10, 0)
            label_bottom = label_top + text_height + baseline + 10
            label_right = min(x1 + text_width + 12, annotated.shape[1] - 1)
            cv2.rectangle(
                annotated,
                (x1, label_top),
                (label_right, label_bottom),
                color,
                thickness=-1,
            )
            cv2.putText(
                annotated,
                caption,
                (x1 + 6, label_bottom - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated
