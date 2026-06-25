from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "gesture_model.pkl"
SCALER_PATH = BASE_DIR / "gesture_scaler.pkl"
ENCODER_PATH = BASE_DIR / "gesture_encoder.pkl"

IMG_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

REFERENCE_FRAME_WIDTH = 720
REFERENCE_FRAME_HEIGHT = 480
REFERENCE_ROI = {
    "top": 50,
    "bottom": 350,
    "left": 300,
    "right": 650,
}


@dataclass(frozen=True)
class ModelBundle:
    model: object
    scaler: object
    encoder: object


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    roi_bgr: np.ndarray | None
    mask: np.ndarray
    segmented_bgr: np.ndarray | None
    prepared_mask: np.ndarray
    foreground_ratio: float
    feature_count: int


def load_artifacts(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    encoder_path: Path = ENCODER_PATH,
) -> ModelBundle:
    with model_path.open("rb") as file:
        model = pickle.load(file)
    with scaler_path.open("rb") as file:
        scaler = pickle.load(file)
    with encoder_path.open("rb") as file:
        encoder = pickle.load(file)
    return ModelBundle(model=model, scaler=scaler, encoder=encoder)


def compute_roi_box(frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    top = round(REFERENCE_ROI["top"] / REFERENCE_FRAME_HEIGHT * height)
    bottom = round(REFERENCE_ROI["bottom"] / REFERENCE_FRAME_HEIGHT * height)
    left = round(REFERENCE_ROI["left"] / REFERENCE_FRAME_WIDTH * width)
    right = round(REFERENCE_ROI["right"] / REFERENCE_FRAME_WIDTH * width)

    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    left = max(0, min(left, width - 1))
    right = max(left + 1, min(right, width))
    return top, bottom, left, right


def preprocess_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top, bottom, left, right = compute_roi_box(frame_bgr.shape)
    roi = frame_bgr[top:bottom, left:right]

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    combined_mask = cv2.bitwise_or(skin_mask, hsv_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    segmented = cv2.bitwise_and(roi, roi, mask=combined_mask)
    return roi, combined_mask, segmented


def prepare_mask(mask: np.ndarray, threshold: bool = False) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    mask = cv2.resize(mask.astype(np.uint8), IMG_SIZE)
    if threshold:
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def extract_hog_features(img: np.ndarray) -> np.ndarray:
    return hog(
        img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=False,
    )


def extract_hu_moments(img: np.ndarray) -> np.ndarray:
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def extract_contour_features(img: np.ndarray) -> np.ndarray:
    features = np.zeros(12)
    contours, _ = cv2.findContours(
        img.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return features

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 100:
        return features

    perimeter = cv2.arcLength(cnt, True)
    _, _, w, h = cv2.boundingRect(cnt)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    solidity = area / (hull_area + 1e-5)
    extent = area / (w * h + 1e-5)
    aspect_ratio = w / (h + 1e-5)
    hull_ratio = hull_perimeter / (perimeter + 1e-5)

    defect_count = 0
    mean_defect_depth = 0.0
    try:
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is not None and len(hull_idx) > 3:
            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is not None:
                for defect in defects:
                    _, _, _, depth = defect[0]
                    depth = depth / 256.0
                    if depth > 10:
                        defect_count += 1
                        mean_defect_depth += depth
                if defect_count > 0:
                    mean_defect_depth /= defect_count
    except cv2.error:
        pass

    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-5)
    equiv_diameter = np.sqrt(4 * area / np.pi)

    return np.array(
        [
            area / (IMG_SIZE[0] * IMG_SIZE[1]),
            perimeter / (2 * (IMG_SIZE[0] + IMG_SIZE[1])),
            aspect_ratio,
            solidity,
            extent,
            hull_ratio,
            circularity,
            equiv_diameter / IMG_SIZE[0],
            defect_count,
            mean_defect_depth / 100.0,
            w / IMG_SIZE[0],
            h / IMG_SIZE[1],
        ]
    )


def extract_pixel_histogram(img: np.ndarray) -> np.ndarray:
    grid = 4
    cell_h = img.shape[0] // grid
    cell_w = img.shape[1] // grid
    features = []

    for row in range(grid):
        for col in range(grid):
            cell = img[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]
            ratio = np.sum(cell > 0) / (cell_h * cell_w + 1e-5)
            features.append(ratio)

    return np.array(features)


def extract_all_features(img: np.ndarray) -> np.ndarray:
    hog_feats = extract_hog_features(img)
    hu_feats = extract_hu_moments(img)
    contour_feats = extract_contour_features(img)
    hist_feats = extract_pixel_histogram(img)
    return np.concatenate([hog_feats, hu_feats, contour_feats, hist_feats])


def predict_from_mask(
    mask: np.ndarray,
    bundle: ModelBundle,
    min_foreground_ratio: float = 0.005,
) -> PredictionResult:
    prepared_mask = prepare_mask(mask, threshold=True)
    foreground_ratio = float(np.mean(prepared_mask > 0))
    if foreground_ratio < min_foreground_ratio:
        raise ValueError("No hand-like foreground was detected in the mask.")

    features = extract_all_features(prepared_mask)
    expected_features = getattr(bundle.scaler, "n_features_in_", features.shape[0])
    if features.shape[0] != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {features.shape[0]}.")

    features_scaled = bundle.scaler.transform([features])
    encoded_prediction = bundle.model.predict(features_scaled)
    label = str(bundle.encoder.inverse_transform(encoded_prediction)[0])

    probabilities: dict[str, float] = {}
    confidence = 1.0
    if hasattr(bundle.model, "predict_proba"):
        proba = bundle.model.predict_proba(features_scaled)[0]
        probabilities = {
            str(bundle.encoder.inverse_transform([encoded_class])[0]): float(probability)
            for encoded_class, probability in zip(bundle.model.classes_, proba)
        }
        confidence = probabilities.get(label, float(np.max(proba)))

    return PredictionResult(
        label=label,
        confidence=float(confidence),
        probabilities=probabilities,
        roi_bgr=None,
        mask=mask,
        segmented_bgr=None,
        prepared_mask=prepared_mask,
        foreground_ratio=foreground_ratio,
        feature_count=int(features.shape[0]),
    )


def predict_from_frame(frame_bgr: np.ndarray, bundle: ModelBundle) -> PredictionResult:
    roi, mask, segmented = preprocess_frame(frame_bgr)
    result = predict_from_mask(mask, bundle)
    return PredictionResult(
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        roi_bgr=roi,
        mask=mask,
        segmented_bgr=segmented,
        prepared_mask=result.prepared_mask,
        foreground_ratio=result.foreground_ratio,
        feature_count=result.feature_count,
    )
