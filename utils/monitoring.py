from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2


@dataclass
class RuntimeConfig:
    min_box_area_ratio: float = 0.0015
    consecutive_frames: int = 3
    alert_cooldown_seconds: float = 2.0
    save_evidence: bool = True
    evidence_dir: str = "evidence"
    overlay_stats: bool = True
    tracking_iou_threshold: float = 0.3
    max_history: int = 10


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    streak: int = 1
    missed: int = 0
    last_confidence: float = 0.0


@dataclass
class EvidenceManager:
    output_dir: Path
    metadata_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "events.jsonl"

    def save(self, frame, event: dict) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = self.output_dir / f"violation_{timestamp}.jpg"
        cv2.imwrite(str(image_path), frame)
        payload = dict(event)
        payload["image_path"] = str(image_path)
        with self.metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return image_path


class ViolationMonitor:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.next_track_id = 1
        self.tracks: dict[int, Track] = {}
        self.confirmed_track_ids: set[int] = set()
        self.recent_events: deque[dict] = deque(maxlen=config.max_history)

    @staticmethod
    def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter_area / float(area_a + area_b - inter_area)

    def _match_track(self, bbox: tuple[int, int, int, int], used_track_ids: set[int]) -> int | None:
        best_track_id = None
        best_iou = 0.0
        for track_id, track in self.tracks.items():
            if track_id in used_track_ids:
                continue
            score = self._iou(track.bbox, bbox)
            if score > best_iou and score >= self.config.tracking_iou_threshold:
                best_iou = score
                best_track_id = track_id
        return best_track_id

    def update(self, no_helmet_detections, frame, timestamp: str, alert_manager, evidence_manager: EvidenceManager | None):
        used_track_ids: set[int] = set()
        confirmed_events: list[dict] = []

        for detection in no_helmet_detections:
            match_id = self._match_track(detection.xyxy, used_track_ids)
            if match_id is None:
                match_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[match_id] = Track(
                    track_id=match_id,
                    bbox=detection.xyxy,
                    streak=1,
                    missed=0,
                    last_confidence=detection.confidence,
                )
            else:
                track = self.tracks[match_id]
                track.bbox = detection.xyxy
                track.streak += 1
                track.missed = 0
                track.last_confidence = detection.confidence

            used_track_ids.add(match_id)
            track = self.tracks[match_id]
            if track.streak >= self.config.consecutive_frames and match_id not in self.confirmed_track_ids:
                self.confirmed_track_ids.add(match_id)
                alert_time = alert_manager.trigger("no_helmet")
                event = {
                    "timestamp": alert_time or timestamp,
                    "track_id": match_id,
                    "confidence": round(track.last_confidence, 4),
                    "label": "no_helmet",
                }
                if evidence_manager is not None:
                    evidence_path = evidence_manager.save(frame, event)
                    event["image_path"] = str(evidence_path)
                confirmed_events.append(event)
                self.recent_events.appendleft(event)

        stale_track_ids: list[int] = []
        for track_id, track in self.tracks.items():
            if track_id in used_track_ids:
                continue
            track.missed += 1
            if track.missed > 10:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            self.tracks.pop(track_id, None)

        return {
            "confirmed_events": confirmed_events,
            "recent_events": list(self.recent_events),
            "active_track_count": len(used_track_ids),
            "unique_violation_count": len(self.confirmed_track_ids),
        }
