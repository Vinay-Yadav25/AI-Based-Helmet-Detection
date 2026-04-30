from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from utils.alerts import AlertManager
from utils.detector import HelmetDetector
from utils.monitoring import EvidenceManager, RuntimeConfig, ViolationMonitor


@dataclass
class DetectionSummary:
    helmet_count: int = 0
    no_helmet_count: int = 0
    violation_count: int = 0
    fps: float = 0.0
    last_alert_time: str | None = None
    timestamp: str = ""
    active_no_helmet_tracks: int = 0
    recent_events: list[dict] | None = None
    latest_evidence_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "helmet_count": self.helmet_count,
            "no_helmet_count": self.no_helmet_count,
            "violation_count": self.violation_count,
            "fps": self.fps,
            "last_alert_time": self.last_alert_time,
            "timestamp": self.timestamp,
            "active_no_helmet_tracks": self.active_no_helmet_tracks,
            "latest_evidence_path": self.latest_evidence_path,
        }


def process_image(
    detector: HelmetDetector,
    frame,
    alert_manager: AlertManager | None = None,
    overlay_stats: bool = True,
    runtime_config: RuntimeConfig | None = None,
    violation_monitor: ViolationMonitor | None = None,
    evidence_manager: EvidenceManager | None = None,
) -> tuple:
    runtime_config = runtime_config or RuntimeConfig()
    detections = detector.predict(frame)
    summary = DetectionSummary(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for detection in detections:
        if detection.label == "helmet":
            summary.helmet_count += 1
        elif detection.label == "no_helmet":
            summary.no_helmet_count += 1

    no_helmet_detections = [detection for detection in detections if detection.label == "no_helmet"]
    if alert_manager and violation_monitor is not None:
        monitor_result = violation_monitor.update(
            no_helmet_detections=no_helmet_detections,
            frame=frame,
            timestamp=summary.timestamp,
            alert_manager=alert_manager,
            evidence_manager=evidence_manager if runtime_config.save_evidence else None,
        )
        summary.violation_count = monitor_result["unique_violation_count"]
        summary.active_no_helmet_tracks = monitor_result["active_track_count"]
        summary.recent_events = monitor_result["recent_events"]
        if monitor_result["confirmed_events"]:
            latest_event = monitor_result["confirmed_events"][0]
            summary.last_alert_time = latest_event["timestamp"]
            summary.latest_evidence_path = latest_event.get("image_path")
        else:
            summary.last_alert_time = (
                monitor_result["recent_events"][0]["timestamp"]
                if monitor_result["recent_events"]
                else None
            )
    elif alert_manager and summary.no_helmet_count > 0:
        summary.last_alert_time = alert_manager.trigger("no_helmet")
        summary.violation_count = alert_manager.total_violations
    elif alert_manager:
        summary.violation_count = alert_manager.total_violations

    annotated = detector.draw_detections(frame, detections)
    if overlay_stats:
        _overlay_runtime_stats(annotated, summary)
    return annotated, summary


def _build_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def process_video_stream(
    detector: HelmetDetector,
    source: int | str,
    save_output: bool = False,
    output_path: Path | None = None,
    window_name: str = "Helmet Detection",
) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    writer = None
    frame_counter = 0
    start_time = time.perf_counter()
    alert_manager = AlertManager(enable_sound=True)

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        source_fps = source_fps if source_fps and source_fps > 0 else 25.0

        if save_output and output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = _build_writer(output_path, width, height, source_fps)

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            annotated, summary = process_image(
                detector=detector,
                frame=frame,
                alert_manager=alert_manager,
            )

            frame_counter += 1
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            summary.fps = frame_counter / elapsed
            if writer:
                writer.write(annotated)

            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def process_video_file(
    detector: HelmetDetector,
    input_path: Path,
    output_path: Path,
    alert_manager: AlertManager | None = None,
    runtime_config: RuntimeConfig | None = None,
    violation_monitor: ViolationMonitor | None = None,
    evidence_manager: EvidenceManager | None = None,
) -> tuple[Path, DetectionSummary]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if source_fps and source_fps > 0 else 25.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _build_writer(output_path, width, height, source_fps)

    runtime_config = runtime_config or RuntimeConfig()
    if alert_manager is not None:
        alert_manager.cooldown_seconds = runtime_config.alert_cooldown_seconds
    if violation_monitor is None:
        violation_monitor = ViolationMonitor(runtime_config)

    summary = DetectionSummary(recent_events=[])
    frame_counter = 0
    start_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            annotated, frame_summary = process_image(
                detector=detector,
                frame=frame,
                alert_manager=alert_manager,
                overlay_stats=runtime_config.overlay_stats,
                runtime_config=runtime_config,
                violation_monitor=violation_monitor,
                evidence_manager=evidence_manager,
            )
            frame_counter += 1

            summary.helmet_count += frame_summary.helmet_count
            summary.no_helmet_count += frame_summary.no_helmet_count
            summary.violation_count = frame_summary.violation_count
            summary.last_alert_time = frame_summary.last_alert_time or summary.last_alert_time
            summary.timestamp = frame_summary.timestamp
            summary.active_no_helmet_tracks = frame_summary.active_no_helmet_tracks
            summary.latest_evidence_path = frame_summary.latest_evidence_path or summary.latest_evidence_path
            summary.recent_events = frame_summary.recent_events or summary.recent_events
            writer.write(annotated)
    finally:
        capture.release()
        writer.release()

    elapsed = max(time.perf_counter() - start_time, 1e-6)
    summary.fps = frame_counter / elapsed
    return output_path, summary


def _overlay_runtime_stats(frame, summary: DetectionSummary) -> None:
    lines = [
        f"Time: {summary.timestamp}",
        f"Helmet: {summary.helmet_count}",
        f"No Helmet: {summary.no_helmet_count}",
        f"Violations: {summary.violation_count}",
        f"FPS: {summary.fps:.2f}",
    ]

    if summary.last_alert_time:
        lines.append(f"Alert: {summary.last_alert_time}")

    max_line_width = 0
    for line in lines:
        (text_width, text_height), baseline = cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        max_line_width = max(max_line_width, text_width)

    panel_height = 12 + len(lines) * 22
    cv2.rectangle(
        frame,
        (5, 5),
        (16 + max_line_width, panel_height),
        (20, 20, 20),
        thickness=-1,
    )

    y = 22
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
