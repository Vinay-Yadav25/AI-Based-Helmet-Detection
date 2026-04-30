from __future__ import annotations

import tempfile
import time
from collections import deque
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from utils.alerts import AlertManager
from utils.detector import HelmetDetector
from utils.monitoring import EvidenceManager, RuntimeConfig, ViolationMonitor
from utils.video import DetectionSummary, process_image, process_video_file

APP_DIR = Path(__file__).resolve().parent
DEFAULT_EVIDENCE_DIR = APP_DIR / "evidence"

def get_default_model_path() -> Path:
    candidates = sorted(
        APP_DIR.rglob("best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return APP_DIR / "runs" / "detect" / "models" / "helmet_detector_stable" / "weights" / "best.pt"


DEFAULT_MODEL = get_default_model_path()

st.set_page_config(page_title="Helmet Detection", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stButton"] {display: none;}
    button[title="Start"],
    button[title="Stop"],
    button[title="Select device"],
    button[title="Settings"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector(model_path: str, conf_threshold: float, min_box_area_ratio: float) -> HelmetDetector:
    return HelmetDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        min_box_area_ratio=min_box_area_ratio,
    )


class LiveVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.detector: HelmetDetector | None = None
        self.alert_manager = AlertManager(enable_sound=False)
        self.runtime_config = RuntimeConfig()
        self.violation_monitor = ViolationMonitor(self.runtime_config)
        self.evidence_manager = EvidenceManager(DEFAULT_EVIDENCE_DIR)
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.perf_counter()
        self.latest_summary = DetectionSummary()
        self.last_alert_time = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        if self.detector is None:
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        processed_frame, summary = process_image(
            detector=self.detector,
            frame=image,
            alert_manager=self.alert_manager,
            runtime_config=self.runtime_config,
            violation_monitor=self.violation_monitor,
            evidence_manager=self.evidence_manager,
        )

        now = time.perf_counter()
        fps = 1.0 / max(now - self.last_frame_time, 1e-6)
        self.last_frame_time = now
        self.fps_history.append(fps)

        summary.fps = sum(self.fps_history) / len(self.fps_history)
        self.latest_summary = summary
        self.last_alert_time = summary.last_alert_time
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")


def sidebar_controls() -> tuple[str, float, RuntimeConfig]:
    st.sidebar.title("Helmet Detection")
    source_mode = st.sidebar.radio(
        "Choose source",
        options=["Upload image", "Upload video", "Webcam mode"],
    )
    conf_threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
    )
    min_box_area_ratio = st.sidebar.slider(
        "Min box area (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.25,
        step=0.05,
        help="Ignore tiny detections smaller than this percentage of the frame area.",
    )
    consecutive_frames = st.sidebar.slider(
        "Violation confirmation frames",
        min_value=1,
        max_value=10,
        value=4,
        help="Require this many consecutive webcam frames before counting a no-helmet violation.",
    )
    alert_cooldown_seconds = st.sidebar.slider(
        "Alert cooldown (seconds)",
        min_value=1,
        max_value=15,
        value=3,
        step=1,
    )
    save_evidence = st.sidebar.checkbox("Save evidence snapshots", value=True)
    overlay_stats = st.sidebar.checkbox("Overlay stats on webcam", value=True)
    runtime_config = RuntimeConfig(
        min_box_area_ratio=min_box_area_ratio / 100.0,
        consecutive_frames=consecutive_frames,
        alert_cooldown_seconds=float(alert_cooldown_seconds),
        save_evidence=save_evidence,
        evidence_dir=str(DEFAULT_EVIDENCE_DIR),
        overlay_stats=overlay_stats,
    )
    return source_mode, conf_threshold, runtime_config


def render_summary(summary: DetectionSummary) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Helmet", summary.helmet_count)
    col2.metric("No Helmet", summary.no_helmet_count)
    col3.metric("Violations", summary.violation_count)
    col4.metric("FPS", f"{summary.fps:.2f}")
    if summary.active_no_helmet_tracks:
        st.caption(f"Active tracked no-helmet workers: {summary.active_no_helmet_tracks}")

    if summary.timestamp:
        st.caption(f"Frame timestamp: {summary.timestamp}")

    if summary.last_alert_time:
        st.warning(f"Alert: no helmet detected at {summary.last_alert_time}")
    else:
        st.success("No active safety violations detected.")

    if summary.latest_evidence_path:
        st.caption(f"Latest evidence: {summary.latest_evidence_path}")


def render_recent_events(summary: DetectionSummary) -> None:
    events = summary.recent_events or []
    st.subheader("Recent Detection History")
    if not events:
        st.info("No confirmed violation events yet.")
        return

    for event in events[:5]:
        details = (
            f"{event['timestamp']} | Track #{event['track_id']} | "
            f"Confidence {event['confidence']:.2f}"
        )
        st.write(details)
        image_path = event.get("image_path")
        if image_path and Path(image_path).exists():
            st.image(image_path, width=220)


def handle_upload_mode(detector: HelmetDetector, enable_sound: bool, runtime_config: RuntimeConfig) -> None:
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if not uploaded:
        st.info("Upload an image to begin detection.")
        return

    suffix = Path(uploaded.name).suffix.lower()
    alert_manager = AlertManager(enable_sound=enable_sound)
    alert_manager.cooldown_seconds = runtime_config.alert_cooldown_seconds
    evidence_manager = EvidenceManager(Path(runtime_config.evidence_dir))

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded.read())
        temp_path = Path(temp_file.name)

    frame = cv2.imread(str(temp_path))
    processed_frame, summary = process_image(
        detector=detector,
        frame=frame,
        alert_manager=alert_manager,
        overlay_stats=False,
        runtime_config=RuntimeConfig(
            **{**runtime_config.__dict__, "consecutive_frames": 1, "overlay_stats": False}
        ),
        violation_monitor=ViolationMonitor(
            RuntimeConfig(
                **{**runtime_config.__dict__, "consecutive_frames": 1, "overlay_stats": False}
            )
        ),
        evidence_manager=evidence_manager,
    )
    st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Detection result")
    render_summary(summary)
    render_recent_events(summary)


def handle_video_upload_mode(
    detector: HelmetDetector,
    enable_sound: bool,
    runtime_config: RuntimeConfig,
) -> None:
    uploaded = st.file_uploader(
        "Upload a CCTV video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if not uploaded:
        st.info("Upload a recorded CCTV video to begin detection.")
        return

    suffix = Path(uploaded.name).suffix.lower()
    alert_manager = AlertManager(enable_sound=enable_sound)
    alert_manager.cooldown_seconds = runtime_config.alert_cooldown_seconds
    video_runtime_config = RuntimeConfig(
        **{**runtime_config.__dict__, "overlay_stats": True}
    )
    violation_monitor = ViolationMonitor(video_runtime_config)
    evidence_manager = EvidenceManager(Path(video_runtime_config.evidence_dir))

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
        temp_input.write(uploaded.read())
        input_path = Path(temp_input.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
        output_path = Path(temp_output.name)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        st.error("Unable to open the uploaded video.")
        return

    frame_placeholder = st.empty()
    summary_placeholder = st.empty()
    history_placeholder = st.empty()
    progress_placeholder = st.empty()

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if source_fps and source_fps > 0 else 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )

    latest_summary = DetectionSummary(recent_events=[])
    frame_index = 0
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
                overlay_stats=video_runtime_config.overlay_stats,
                runtime_config=video_runtime_config,
                violation_monitor=violation_monitor,
                evidence_manager=evidence_manager,
            )

            frame_index += 1
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            frame_summary.fps = min(source_fps, frame_index / elapsed)
            latest_summary = frame_summary

            writer.write(annotated)
            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                caption="Uploaded CCTV video detection",
                use_container_width=True,
            )
            with summary_placeholder.container():
                render_summary(latest_summary)
            with history_placeholder.container():
                render_recent_events(latest_summary)

            if total_frames > 0:
                progress_placeholder.progress(min(frame_index / total_frames, 1.0))

        latest_summary.fps = source_fps
    finally:
        capture.release()
        writer.release()

    st.success("Video detection completed.")
    st.video(output_path.read_bytes())


def handle_webcam_mode(detector: HelmetDetector, enable_sound: bool, runtime_config: RuntimeConfig) -> None:
    st.info("Allow camera access in your browser. Detection starts automatically in webcam mode.")
    summary_placeholder = st.empty()
    history_placeholder = st.empty()
    ctx = webrtc_streamer(
        key="helmet-webcam",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=LiveVideoProcessor,
        async_processing=True,
        desired_playing_state=True,
    )

    if ctx.video_processor:
        ctx.video_processor.detector = detector
        ctx.video_processor.alert_manager.enable_sound = enable_sound
        ctx.video_processor.alert_manager.cooldown_seconds = runtime_config.alert_cooldown_seconds
        ctx.video_processor.runtime_config = runtime_config
        ctx.video_processor.violation_monitor = ViolationMonitor(runtime_config)
        ctx.video_processor.evidence_manager = EvidenceManager(Path(runtime_config.evidence_dir))

        while ctx.state.playing:
            with summary_placeholder:
                render_summary(ctx.video_processor.latest_summary)
            with history_placeholder:
                render_recent_events(ctx.video_processor.latest_summary)
            time.sleep(0.5)
    else:
        summary_placeholder.info("Webcam session is idle.")


def main() -> None:
    st.title("Real-Time Safety Helmet Detection")
    st.caption("YOLOv8 + OpenCV + Streamlit")

    source_mode, conf_threshold, runtime_config = sidebar_controls()
    enable_sound = st.sidebar.checkbox("Enable alert sound", value=False)
    model_path = st.sidebar.text_input("Model path", value=str(DEFAULT_MODEL))

    try:
        detector = load_detector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            min_box_area_ratio=runtime_config.min_box_area_ratio,
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train a model first or update the model path in the sidebar.")
        st.stop()

    if source_mode == "Upload image":
        handle_upload_mode(
            detector=detector,
            enable_sound=enable_sound,
            runtime_config=runtime_config,
        )
    elif source_mode == "Upload video":
        handle_video_upload_mode(
            detector=detector,
            enable_sound=enable_sound,
            runtime_config=runtime_config,
        )
    else:
        handle_webcam_mode(
            detector=detector,
            enable_sound=enable_sound,
            runtime_config=runtime_config,
        )


if __name__ == "__main__":
    main()
