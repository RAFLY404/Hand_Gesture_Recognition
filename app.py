from __future__ import annotations

import threading
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from gesture_predictor import compute_roi_box, load_artifacts, predict_from_frame


st.set_page_config(
    page_title="Live Hand Gesture Recognition",
    layout="wide",
    initial_sidebar_state="expanded",
)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def prettify_label(label: str) -> str:
    return label.replace("_", " ").title()


def draw_label_box(frame_bgr: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame_bgr, (16, 16), (520, 74), (15, 23, 42), -1)
    cv2.putText(
        frame_bgr,
        text,
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_mask_inset(frame_bgr: np.ndarray, mask: np.ndarray | None) -> None:
    if mask is None:
        return

    inset_size = 150
    mask_small = cv2.resize(mask, (inset_size, inset_size))
    mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
    y1, x1 = 92, 16
    y2, x2 = y1 + inset_size, x1 + inset_size

    if y2 > frame_bgr.shape[0] or x2 > frame_bgr.shape[1]:
        return

    frame_bgr[y1:y2, x1:x2] = mask_bgr
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(
        frame_bgr,
        "Mask",
        (x1 + 8, y2 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


class GestureVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.bundle = load_artifacts()
        self.lock = threading.Lock()
        self.frame_index = 0
        self.label = "Waiting"
        self.confidence = 0.0
        self.probabilities: dict[str, float] = {}
        self.foreground_ratio = 0.0
        self.feature_count = 0
        self.error = ""
        self.latest_mask: np.ndarray | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_bgr = cv2.flip(frame_bgr, 1)
        self.frame_index += 1

        top, bottom, left, right = compute_roi_box(frame_bgr.shape)
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), (34, 197, 94), 3)

        if self.frame_index % 3 == 0:
            try:
                result = predict_from_frame(frame_bgr, self.bundle)
                with self.lock:
                    self.label = result.label
                    self.confidence = result.confidence
                    self.probabilities = result.probabilities
                    self.foreground_ratio = result.foreground_ratio
                    self.feature_count = result.feature_count
                    self.error = ""
                    self.latest_mask = result.prepared_mask
            except Exception as exc:
                with self.lock:
                    self.label = "No hand"
                    self.confidence = 0.0
                    self.probabilities = {}
                    self.foreground_ratio = 0.0
                    self.feature_count = 0
                    self.error = str(exc)
                    self.latest_mask = None

        with self.lock:
            label = self.label
            confidence = self.confidence
            latest_mask = None if self.latest_mask is None else self.latest_mask.copy()

        if label == "No hand":
            draw_label_box(frame_bgr, "Prediction: No hand detected", (96, 165, 250))
        elif label == "Waiting":
            draw_label_box(frame_bgr, "Prediction: Waiting for video", (96, 165, 250))
        else:
            text = f"Prediction: {prettify_label(label)} ({confidence:.1%})"
            draw_label_box(frame_bgr, text, (74, 222, 128))

        draw_mask_inset(frame_bgr, latest_mask)
        return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")


st.title("Live Hand Gesture Recognition")
st.caption("Allow camera access, place your hand inside the green box, and keep the background simple.")

with st.sidebar:
    st.header("Model")
    st.write("Gestures: fist, open hand, peace, pointing, thumbs up.")
    st.write("Prediction runs on every third video frame to keep the webcam stream smooth.")
    st.divider()
    st.header("Camera Tips")
    st.write("Use good lighting.")
    st.write("Keep your hand inside the green region.")
    st.write("Use a plain background when possible.")

left, right = st.columns([1.35, 0.65], gap="large")

with left:
    ctx = webrtc_streamer(
        key="live-gesture-recognition",
        video_processor_factory=GestureVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with right:
    st.subheader("Live Status")
    status_box = st.empty()
    score_box = st.empty()

if ctx.video_processor:
    while ctx.state.playing:
        with ctx.video_processor.lock:
            label = ctx.video_processor.label
            confidence = ctx.video_processor.confidence
            probabilities = dict(ctx.video_processor.probabilities)
            foreground_ratio = ctx.video_processor.foreground_ratio
            feature_count = ctx.video_processor.feature_count
            error = ctx.video_processor.error

        with status_box.container():
            if label in {"Waiting", "No hand"}:
                st.metric("Gesture", label)
            else:
                st.metric("Gesture", prettify_label(label))
            st.metric("Confidence", f"{confidence:.1%}")
            st.caption(f"Foreground ratio: {foreground_ratio:.2%}")
            if feature_count:
                st.caption(f"Feature vector: {feature_count} values")
            if error:
                st.caption(error)

        with score_box.container():
            if probabilities:
                st.divider()
                st.subheader("Class Scores")
                for class_label, probability in sorted(
                    probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                ):
                    st.write(prettify_label(class_label))
                    st.progress(probability, text=f"{probability:.1%}")

        time.sleep(0.4)
else:
    with status_box.container():
        st.info("Start the camera stream to begin live detection.")
