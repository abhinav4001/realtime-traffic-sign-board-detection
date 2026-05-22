import json
import re
import sys
from pathlib import Path

import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_SIZE = (224, 224)
MIN_CONTOUR_AREA = 250
MIN_BLUE_CONTOUR_AREA = 120
CONFIDENCE_THRESHOLD = 70.0
MIN_PREDICTION_CONFIDENCE = 60.0
MIN_TOP_GAP = 8.0
MIN_BOX_SIDE = 24
MAX_BOX_SIDE_RATIO = 0.75
MAX_BOX_AREA_RATIO = 0.55
MIN_CIRCLE_FILL_RATIO_RED = 0.38
MIN_CIRCLE_FILL_RATIO_BLUE = 0.30
CLASS_ORDER_PATH = BASE_DIR / "class_order.json"
SPEED_TEMPLATE_SIZE = (32, 48)
SPEED_SIGN_THRESHOLDS = (80, 90, 100, 110, 120, 130, 140)
TEMPLATE_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
)
MAX_SIGN_CANDIDATES = 8
OUTPUT_DIR = BASE_DIR / "detected_sign_crops"
DEBUG_OUTPUT_DIR = BASE_DIR / "detection_debug"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model(BASE_DIR / "traffic_sign_model.keras")

# -----------------------------
# Load label names
# -----------------------------
labels_df = pd.read_csv(BASE_DIR / "labels.csv")

def build_class_mapping(labels_dataframe, class_order_path):
    names_by_class_id = {
        str(class_id): name
        for class_id, name in zip(labels_dataframe.ClassId, labels_dataframe.Name)
    }

    if class_order_path.exists():
        with class_order_path.open("r", encoding="utf-8") as file:
            class_order = json.load(file)
    else:
        # Fallback for older models trained with image_dataset_from_directory().
        class_order = sorted(names_by_class_id.keys(), key=str)

    return {
        model_index: names_by_class_id[str(class_id)]
        for model_index, class_id in enumerate(class_order)
    }


class_names = build_class_mapping(labels_df, CLASS_ORDER_PATH)
SPEED_LIMIT_PATTERN = re.compile(r"speed limit \((\d+)km/h\)", re.IGNORECASE)
KNOWN_SPEED_LIMITS = sorted(
    {
        extract.group(1)
        for label in class_names.values()
        if (extract := SPEED_LIMIT_PATTERN.search(label))
    },
    key=int,
)


def render_digit_template(digit, font, size=SPEED_TEMPLATE_SIZE):
    width, height = size
    image = np.zeros((height, width), dtype=np.uint8)
    scale = 1.5
    thickness = 3
    (text_width, text_height), _ = cv2.getTextSize(str(digit), font, scale, thickness)
    origin = ((width - text_width) // 2, (height + text_height) // 2 - 2)
    cv2.putText(
        image,
        str(digit),
        origin,
        font,
        scale,
        255,
        thickness,
        cv2.LINE_AA,
    )
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image


def digit_shape_features(binary_digit):
    contours, hierarchy = cv2.findContours(
        binary_digit, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = max(cv2.contourArea(hull), 1.0)

    holes = 0
    if hierarchy is not None:
        for item in hierarchy[0]:
            if item[3] != -1:
                holes += 1

    moments = cv2.moments(contour)
    center_x = moments["m10"] / moments["m00"] if moments["m00"] else w / 2.0
    center_y = moments["m01"] / moments["m00"] if moments["m00"] else h / 2.0

    return {
        "area_ratio": area / max(w * h, 1),
        "solidity": area / hull_area,
        "aspect_ratio": w / max(float(h), 1.0),
        "holes": holes,
        "center_x": center_x / max(w, 1),
        "center_y": center_y / max(h, 1),
    }


def build_digit_templates():
    templates = {}
    for digit in map(str, range(10)):
        templates[digit] = []
        for font in TEMPLATE_FONTS:
            template = render_digit_template(digit, font)
            features = digit_shape_features(template)
            if features is not None:
                templates[digit].append({"features": features, "image": template})
    return templates


DIGIT_TEMPLATES = build_digit_templates()

# -----------------------------
# CNN classification
# -----------------------------
def classify_sign(crop):
    img = cv2.resize(crop, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]

    class_id = int(np.argmax(pred))
    confidence = float(pred[class_id] * 100)
    top2_index = int(np.argsort(pred)[-2]) if len(pred) > 1 else class_id
    top2_confidence = float(pred[top2_index] * 100)

    return (
        class_names.get(class_id, f"Class {class_id}"),
        confidence,
        class_names.get(top2_index, f"Class {top2_index}"),
        top2_confidence,
    )


def center_crop(frame, crop_ratio=0.72):
    height, width = frame.shape[:2]
    crop_size = int(min(height, width) * crop_ratio)
    x = max(0, (width - crop_size) // 2)
    y = max(0, (height - crop_size) // 2)
    return frame[y : y + crop_size, x : x + crop_size], (x, y, crop_size, crop_size)


def crop_from_box(frame, box):
    x, y, w, h = box
    return frame[y : y + h, x : x + w]


def find_dominant_speed_sign_circle(
    frame,
    param2=26,
    min_radius_ratio=0.09,
    max_radius_ratio=0.42,
    min_red_ratio=0.22,
    min_white_ratio=0.30,
    min_dark_ratio=0.03,
):
    height, width = frame.shape[:2]
    min_side = min(height, width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, int(min_side * 0.18)),
        param1=120,
        param2=param2,
        minRadius=max(14, int(min_side * min_radius_ratio)),
        maxRadius=max(28, int(min_side * max_radius_ratio)),
    )

    if circles is None:
        return None

    red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 70, 255]))
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 110]))

    best = None

    for cx, cy, radius in np.round(circles[0]).astype(int):
        if radius < 14:
            continue

        ring_mask = np.zeros((height, width), dtype=np.uint8)
        inner_mask = np.zeros((height, width), dtype=np.uint8)
        digit_mask = np.zeros((height, width), dtype=np.uint8)

        cv2.circle(ring_mask, (cx, cy), int(radius * 1.02), 255, -1)
        cv2.circle(ring_mask, (cx, cy), int(radius * 0.66), 0, -1)
        cv2.circle(inner_mask, (cx, cy), int(radius * 0.62), 255, -1)
        cv2.circle(digit_mask, (cx, cy), int(radius * 0.52), 255, -1)

        ring_pixels = float(max(np.count_nonzero(ring_mask), 1))
        inner_pixels = float(max(np.count_nonzero(inner_mask), 1))
        digit_pixels = float(max(np.count_nonzero(digit_mask), 1))

        red_ratio = np.count_nonzero(cv2.bitwise_and(red_mask, ring_mask)) / ring_pixels
        white_ratio = np.count_nonzero(cv2.bitwise_and(white_mask, inner_mask)) / inner_pixels
        dark_ratio = np.count_nonzero(cv2.bitwise_and(dark_mask, digit_mask)) / digit_pixels

        if red_ratio < min_red_ratio or white_ratio < min_white_ratio or dark_ratio < min_dark_ratio:
            continue

        x1 = max(0, int(cx - radius * 1.18))
        y1 = max(0, int(cy - radius * 1.18))
        x2 = min(width, int(cx + radius * 1.18))
        y2 = min(height, int(cy + radius * 1.18))
        box = (x1, y1, x2 - x1, y2 - y1)
        center_bonus = 1.0 - (
            (abs(cx - width / 2.0) / max(width / 2.0, 1.0))
            + (abs(cy - height / 2.0) / max(height / 2.0, 1.0))
        ) / 2.0
        size_bonus = radius / float(max(min_side, 1))
        score = (
            (red_ratio * 3.2)
            + (white_ratio * 2.1)
            + (dark_ratio * 1.8)
            + (center_bonus * 0.8)
            + (size_bonus * 1.4)
        )

        candidate = {"box": box, "circle": (cx, cy, radius), "score": score}
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def find_dominant_speed_sign_circle_relaxed(frame):
    return find_dominant_speed_sign_circle(
        frame,
        param2=20,
        min_radius_ratio=0.06,
        max_radius_ratio=0.48,
        min_red_ratio=0.15,
        min_white_ratio=0.22,
        min_dark_ratio=0.02,
    )


def find_dominant_speed_sign_circle_scaled(frame, min_side=220):
    height, width = frame.shape[:2]
    smallest_side = min(height, width)
    if smallest_side == 0:
        return None

    if smallest_side < min_side:
        scale = min_side / float(smallest_side)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        candidate = find_dominant_speed_sign_circle(resized)
        if candidate is None:
            candidate = find_dominant_speed_sign_circle_relaxed(resized)
        if candidate is None:
            return None
        x, y, w, h = candidate["box"]
        box = (
            int(round(x / scale)),
            int(round(y / scale)),
            int(round(w / scale)),
            int(round(h / scale)),
        )
        cx, cy, radius = candidate["circle"]
        circle = (
            int(round(cx / scale)),
            int(round(cy / scale)),
            int(round(radius / scale)),
        )
        return {"box": box, "circle": circle, "score": candidate["score"]}

    candidate = find_dominant_speed_sign_circle(frame)
    if candidate is None:
        candidate = find_dominant_speed_sign_circle_relaxed(frame)
    return candidate


def extract_speed_limit_from_circle_crop(crop, circle=None):
    if circle is None:
        candidate = find_dominant_speed_sign_circle(crop)
        if candidate is None:
            return None, None
        circle = candidate["circle"]

    center_x, center_y, radius = circle
    if radius < 12:
        return None, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    inner_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(inner_mask, (int(center_x), int(center_y)), int(radius * 0.60), 255, -1)

    kernel = np.ones((2, 2), np.uint8)
    best_value = None
    best_score = float("inf")
    expected_center_x = center_x
    expected_center_y = center_y
    min_digit_height = max(10, int(radius * 0.28))
    max_digit_height = int(radius * 1.05)
    max_center_y_offset = radius * 0.28

    for threshold in SPEED_SIGN_THRESHOLDS:
        binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        binary = cv2.bitwise_and(binary, inner_mask)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        digit_contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []

        for digit_contour in digit_contours:
            x, y, w, h = cv2.boundingRect(digit_contour)
            area = cv2.contourArea(digit_contour)
            if area < 6 or area > crop.shape[0] * crop.shape[1] * 0.18:
                continue
            if h < min_digit_height or h > max_digit_height:
                continue
            if not 0.14 <= w / float(max(h, 1)) <= 1.1:
                continue
            box_center_x = x + (w / 2.0)
            box_center_y = y + (h / 2.0)
            if abs(box_center_y - expected_center_y) > max_center_y_offset:
                continue
            if abs(box_center_x - expected_center_x) > radius * 0.62:
                continue
            score = (
                h * 2.2
                - abs(box_center_y - expected_center_y) * 1.8
                - abs(box_center_x - expected_center_x) * 0.35
            )
            boxes.append((x, y, w, h, score))

        boxes = sorted(boxes, key=lambda box: box[4], reverse=True)
        if len(boxes) >= 2:
            boxes = sorted(boxes[:2], key=lambda box: box[0])
        elif len(boxes) == 1:
            boxes = [boxes[0]]
        else:
            continue

        digit_features = []
        for x, y, w, h, _ in boxes:
            digit_image = binary[y : y + h, x : x + w]
            digit_image = cv2.resize(
                digit_image, SPEED_TEMPLATE_SIZE, interpolation=cv2.INTER_NEAREST
            )
            features = digit_shape_features(digit_image)
            if features is None:
                digit_features = []
                break
            digit_features.append((digit_image, features))

        if not digit_features:
            continue

        for candidate in KNOWN_SPEED_LIMITS:
            if len(candidate) != len(digit_features):
                continue

            score = sum(
                score_digit_features(digit_image, features, digit)
                for (digit_image, features), digit in zip(digit_features, candidate)
            )

            if score < best_score:
                best_score = score
                best_value = candidate

    if best_value is None:
        return None, None

    confidence = max(0.0, min(99.0, 100.0 - (best_score * 35.0)))
    return best_value, confidence


def analyze_simple_candidate(crop, box):
    name, confidence, secondary_name, secondary_confidence = classify_sign(crop)
    speed_value, speed_confidence = extract_speed_limit_from_circle_crop(crop)
    name, confidence = refine_prediction_with_speed_value(
        crop,
        name,
        confidence,
        secondary_name,
        secondary_confidence,
        speed_value,
        speed_confidence,
    )
    return {
        "box": box,
        "name": name,
        "confidence": confidence,
        "secondary_name": secondary_name,
        "secondary_confidence": secondary_confidence,
        "ocr_speed": speed_value,
        "ocr_confidence": speed_confidence,
    }


def analyze_dataset_image(frame):
    full_box = (0, 0, frame.shape[1], frame.shape[0])
    full_result = analyze_simple_candidate(frame, full_box)

    speed_candidate = find_dominant_speed_sign_circle_scaled(frame)
    if speed_candidate is not None:
        speed_crop = crop_from_box(frame, speed_candidate["box"])
        direct_speed_value, direct_speed_confidence = extract_speed_limit_from_circle_crop(
            speed_crop
        )
        if direct_speed_value is not None and direct_speed_confidence is not None:
            return {
                "box": speed_candidate["box"],
                "name": f"Speed limit ({direct_speed_value}km/h)",
                "confidence": max(82.0, direct_speed_confidence),
                "secondary_name": full_result["name"],
                "secondary_confidence": full_result["confidence"],
                "ocr_speed": direct_speed_value,
                "ocr_confidence": direct_speed_confidence,
            }

        speed_result = analyze_simple_candidate(speed_crop, speed_candidate["box"])
        if is_speed_limit_label(speed_result["name"]):
            speed_result["confidence"] = max(speed_result["confidence"], 90.0)
            if not is_speed_limit_label(full_result["name"]):
                return speed_result

            if speed_result["name"] != full_result["name"]:
                return speed_result

            full_result["confidence"] = max(full_result["confidence"], 85.0)
            return speed_result if speed_result["confidence"] >= full_result["confidence"] else full_result

    if is_speed_limit_label(full_result["name"]):
        full_result["confidence"] = max(full_result["confidence"], 85.0)
        return full_result

    if full_result["confidence"] >= 72.0:
        return full_result

    crop, crop_box = center_crop(frame)
    crop_result = analyze_simple_candidate(crop, crop_box)
    if crop_result["confidence"] >= full_result["confidence"] + 6.0:
        return crop_result

    return full_result


def _apply_gamma(frame, gamma):
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(frame, table)


def analyze_dataset_image_retry(frame):
    # Try a few deterministic variants to help digit extraction on hard images.
    variants = [
        ("orig", frame),
        ("contrast_up", cv2.convertScaleAbs(frame, alpha=1.15, beta=0)),
        ("contrast_down", cv2.convertScaleAbs(frame, alpha=0.9, beta=0)),
        ("bright", cv2.convertScaleAbs(frame, alpha=1.0, beta=12)),
        ("gamma_up", _apply_gamma(frame, 1.2)),
        ("gamma_down", _apply_gamma(frame, 0.85)),
    ]

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    variants.append(("sharpen", cv2.filter2D(frame, -1, kernel)))

    best = None
    best_score = -1.0

    for _, variant in variants:
        result = analyze_dataset_image(variant)
        score = result["confidence"]
        if is_speed_limit_label(result["name"]):
            score += 12.0
        if score > best_score:
            best_score = score
            best = result

    return best if best is not None else analyze_dataset_image(frame)

def analyze_simple_image(frame):
    candidates = [
        analyze_simple_candidate(frame, (0, 0, frame.shape[1], frame.shape[0]))
    ]

    crop, crop_box = center_crop(frame)
    candidates.append(analyze_simple_candidate(crop, crop_box))

    speed_candidate = find_dominant_speed_sign_circle_scaled(frame)
    if speed_candidate is not None:
        speed_crop = crop_from_box(frame, speed_candidate["box"])
        speed_result = analyze_simple_candidate(speed_crop, speed_candidate["box"])
        if is_speed_limit_label(speed_result["name"]):
            speed_result["confidence"] = max(speed_result["confidence"], 88.0)
        else:
            speed_result["confidence"] += 6.0
        candidates.append(speed_result)

    return max(candidates, key=lambda item: item["confidence"])


def extract_speed_limit_value(label):
    match = SPEED_LIMIT_PATTERN.search(label)
    return match.group(1) if match else None


def is_speed_limit_label(label):
    return extract_speed_limit_value(label) is not None


def score_digit_features(binary_digit, features, digit):
    best_score = float("inf")

    for template in DIGIT_TEMPLATES[digit]:
        template_features = template["features"]
        score = 0.0
        score += abs(features["area_ratio"] - template_features["area_ratio"]) * 3.0
        score += abs(features["solidity"] - template_features["solidity"]) * 3.0
        score += abs(features["aspect_ratio"] - template_features["aspect_ratio"]) * 2.0
        score += abs(features["center_x"] - template_features["center_x"]) * 1.5
        score += abs(features["center_y"] - template_features["center_y"]) * 1.5
        score += abs(features["holes"] - template_features["holes"]) * 2.0

        # Pixel overlap helps separate similar silhouettes like 4/7 and 5/8.
        template_image = template["image"]
        xor_ratio = np.count_nonzero(cv2.bitwise_xor(binary_digit, template_image)) / float(
            max(binary_digit.size, 1)
        )
        score += xor_ratio * 6.0
        best_score = min(best_score, score)

    return best_score


def extract_speed_limit_from_crop(crop):
    mask = build_color_mask(crop)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
        return None, None

    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    if radius < 12:
        return None, None

    inner_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(
        inner_mask,
        (int(center_x), int(center_y)),
        int(radius * 0.68),
        255,
        -1,
    )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    kernel = np.ones((2, 2), np.uint8)

    best_value = None
    best_score = float("inf")

    for threshold in SPEED_SIGN_THRESHOLDS:
        binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        binary = cv2.bitwise_and(binary, inner_mask)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        digit_contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []

        for digit_contour in digit_contours:
            x, y, w, h = cv2.boundingRect(digit_contour)
            area = cv2.contourArea(digit_contour)
            if area < 5 or area > crop.shape[0] * crop.shape[1] * 0.3:
                continue
            if h < max(10, int(radius * 0.2)):
                continue
            if not 0.15 <= w / float(h) <= 1.2:
                continue
            boxes.append((x, y, w, h))

        boxes = sorted(boxes, key=lambda box: box[0])[:2]
        if not boxes:
            continue

        digit_features = []
        for x, y, w, h in boxes:
            digit_image = binary[y : y + h, x : x + w]
            digit_image = cv2.resize(
                digit_image, SPEED_TEMPLATE_SIZE, interpolation=cv2.INTER_NEAREST
            )
            features = digit_shape_features(digit_image)
            if features is None:
                digit_features = []
                break
            digit_features.append(features)

        if not digit_features:
            continue

        for candidate in KNOWN_SPEED_LIMITS:
            if len(candidate) != len(digit_features):
                continue

            score = sum(
                score_digit_features(features, digit)
                for features, digit in zip(digit_features, candidate)
            )

            if score < best_score:
                best_score = score
                best_value = candidate

    if best_value is None:
        return None, None

    confidence = max(0.0, min(99.0, 100.0 - (best_score * 35.0)))
    return best_value, confidence


def build_display_label(name, confidence):
    speed_value = extract_speed_limit_value(name)
    if speed_value is not None:
        return f"Speed limit: {speed_value} km/h ({confidence:.1f}%)"
    return f"{name} {confidence:.1f}%"


def choose_prediction(primary_name, primary_confidence, secondary_name, secondary_confidence):
    return primary_name, primary_confidence


def crop_looks_like_speed_sign(crop):
    return find_dominant_speed_sign_circle(crop) is not None


def refine_prediction_with_speed_value(
    crop,
    primary_name,
    primary_confidence,
    secondary_name,
    secondary_confidence,
    detected_speed_value,
    detected_speed_confidence,
):
    if detected_speed_value is None:
        return choose_prediction(
            primary_name, primary_confidence, secondary_name, secondary_confidence
        )

    speed_label = f"Speed limit ({detected_speed_value}km/h)"
    primary_speed = extract_speed_limit_value(primary_name)
    secondary_speed = extract_speed_limit_value(secondary_name)
    speed_like_crop = crop_looks_like_speed_sign(crop)

    if not speed_like_crop:
        return choose_prediction(
            primary_name, primary_confidence, secondary_name, secondary_confidence
        )

    if primary_speed == detected_speed_value:
        return primary_name, max(primary_confidence, detected_speed_confidence)

    if secondary_speed == detected_speed_value and secondary_confidence >= primary_confidence + 5:
        return secondary_name, max(secondary_confidence, detected_speed_confidence)

    if primary_speed is not None:
        if detected_speed_confidence >= 68.0:
            return speed_label, max(detected_speed_confidence, primary_confidence + 1.0)
        return primary_name, max(primary_confidence, detected_speed_confidence)

    if detected_speed_confidence >= 75.0 and primary_confidence < 70.0:
        return speed_label, detected_speed_confidence

    return choose_prediction(
        primary_name, primary_confidence, secondary_name, secondary_confidence
    )


def prediction_is_reliable(
    primary_name,
    primary_confidence,
    secondary_name,
    secondary_confidence,
    speed_value,
    speed_confidence,
    candidate_score,
    symbol_score,
):
    if (
        is_speed_limit_label(primary_name)
        and speed_value is not None
        and speed_confidence is not None
        and speed_confidence >= 65.0
    ):
        return True

    if symbol_score >= 0.03 and primary_confidence >= 50.0:
        return True

    if candidate_score >= 2500 and primary_confidence >= 50.0:
        return True

    if primary_confidence < MIN_PREDICTION_CONFIDENCE:
        return False

    if (primary_confidence - secondary_confidence) < MIN_TOP_GAP:
        return False

    return True


# -----------------------------
# Detect traffic sign using color mask
# -----------------------------
def build_color_masks(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Keep the ranges broad enough for webcam use, but tighter on blue so
    # glass reflections from buildings are less likely to dominate.
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([95, 80, 60])
    upper_blue = np.array([135, 255, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 85])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(
        hsv, lower_red2, upper_red2
    )
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    black_mask = cv2.bitwise_and(
        black_mask,
        cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)[1],
    )

    red_kernel = np.ones((5, 5), np.uint8)
    blue_kernel = np.ones((3, 3), np.uint8)
    black_kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, red_kernel)
    red_mask = cv2.dilate(red_mask, red_kernel, iterations=1)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, blue_kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, blue_kernel)
    blue_mask = cv2.dilate(blue_mask, blue_kernel, iterations=1)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, black_kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, black_kernel)

    combined_mask = red_mask | blue_mask
    return red_mask, blue_mask, black_mask, combined_mask


def build_color_mask(frame):
    _, _, _, combined_mask = build_color_masks(frame)
    return combined_mask


def colorize_mask(mask, color):
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored[mask > 0] = color
    return colored


def resize_for_preview(image, width=500):
    height, current_width = image.shape[:2]
    if current_width <= width:
        return image

    scale = width / float(current_width)
    new_height = max(1, int(height * scale))
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)


def build_mask_preview(frame):
    red_mask, blue_mask, black_mask, combined_mask = build_color_masks(frame)
    red_preview = colorize_mask(red_mask, (0, 0, 255))
    blue_preview = colorize_mask(blue_mask, (255, 0, 0))
    black_preview = colorize_mask(black_mask, (80, 80, 80))
    combined_preview = colorize_mask(combined_mask, (255, 255, 255))

    top_row = np.hstack(
        [
            resize_for_preview(frame),
            resize_for_preview(red_preview),
        ]
    )
    bottom_row = np.hstack(
        [
            resize_for_preview(blue_preview),
            resize_for_preview(black_preview),
        ]
    )
    third_row = np.hstack(
        [
            resize_for_preview(combined_preview),
            resize_for_preview(frame),
        ]
    )
    preview = np.vstack([top_row, bottom_row, third_row])

    tile_width = top_row.shape[1] // 2
    tile_height = top_row.shape[0]
    labels = (
        ("Original", 10, 30),
        ("Red Mask", tile_width + 10, 30),
        ("Blue Mask", 10, tile_height + 30),
        ("Black Mask", tile_width + 10, tile_height + 30),
        ("Combined Mask", 10, (tile_height * 2) + 30),
    )
    for text, x, y in labels:
        cv2.putText(
            preview,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    return preview, combined_mask


def black_symbol_score(frame, box):
    _, _, black_mask, _ = build_color_masks(frame)
    x, y, w, h = box

    inner_x1 = x + int(w * 0.2)
    inner_y1 = y + int(h * 0.2)
    inner_x2 = x + int(w * 0.8)
    inner_y2 = y + int(h * 0.8)

    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        return 0.0

    inner_mask = black_mask[inner_y1:inner_y2, inner_x1:inner_x2]
    if inner_mask.size == 0:
        return 0.0

    black_ratio = np.count_nonzero(inner_mask) / float(inner_mask.size)
    return black_ratio


def refine_sign_box_from_color_mask(frame, box, color_name):
    red_mask, blue_mask, _, _ = build_color_masks(frame)
    color_mask = red_mask if color_name == "red" else blue_mask

    x, y, w, h = box
    roi = color_mask[y : y + h, x : x + w]
    if roi.size == 0:
        return box

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return box

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return box

    contour = contour + np.array([[[x, y]]], dtype=np.int32)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    if radius < 12:
        return box

    circle_area = np.pi * radius * radius
    fill_ratio = area / float(max(circle_area, 1.0))
    min_fill_ratio = (
        MIN_CIRCLE_FILL_RATIO_RED if color_name == "red" else MIN_CIRCLE_FILL_RATIO_BLUE
    )
    if fill_ratio < min_fill_ratio:
        return box

    pad = int(radius * 0.18)
    x1 = max(0, int(cx - radius) - pad)
    y1 = max(0, int(cy - radius) - pad)
    x2 = min(frame.shape[1], int(cx + radius) + pad)
    y2 = min(frame.shape[0], int(cy + radius) + pad)
    return (x1, y1, x2 - x1, y2 - y1)


def score_sign_contour(contour, frame_shape, color_name):
    area = cv2.contourArea(contour)
    min_area = MIN_BLUE_CONTOUR_AREA if color_name == "blue" else MIN_CONTOUR_AREA
    if area < min_area:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None

    frame_height, frame_width = frame_shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return None

    if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
        return None

    if x <= 1 or y <= 1 or x + w >= frame_width - 1 or y + h >= frame_height - 1:
        return None

    aspect_ratio = w / float(h)
    if not 0.45 <= aspect_ratio <= 1.45:
        return None

    box_area = w * h
    frame_area = frame_width * frame_height
    if box_area > frame_area * MAX_BOX_AREA_RATIO:
        return None

    if max(w, h) > min(frame_width, frame_height) * MAX_BOX_SIDE_RATIO:
        return None

    # Small edge fragments from posters, reflections, or screen borders often
    # trigger the mask. Reject them aggressively if they hug the frame edges.
    if box_area < frame_area * 0.01:
        near_left_or_right_edge = x < frame_width * 0.08 or (x + w) > frame_width * 0.92
        near_top_or_bottom_edge = y < frame_height * 0.08 or (y + h) > frame_height * 0.92
        if near_left_or_right_edge or near_top_or_bottom_edge:
            return None

    circularity = 4 * np.pi * area / (perimeter * perimeter)
    fill_ratio = area / float(max(box_area, 1))
    (_, _), radius = cv2.minEnclosingCircle(contour)
    circle_fill_ratio = area / float(max(np.pi * radius * radius, 1.0))
    center_y = y + (h / 2.0)
    vertical_position = center_y / float(max(frame_height, 1))

    if circularity < 0.2 or fill_ratio < 0.18:
        return None

    if color_name == "blue":
        if circle_fill_ratio < MIN_CIRCLE_FILL_RATIO_BLUE:
            return None
    elif circle_fill_ratio < MIN_CIRCLE_FILL_RATIO_RED:
        return None

    score = (
        area
        + (circularity * 1200)
        + (fill_ratio * 500)
        + (circle_fill_ratio * 700)
        + (vertical_position * 450)
    )
    if color_name == "red":
        score += 250
    if color_name == "blue":
        score += 150
    return score, (x, y, w, h)


def expand_box(box, frame_shape, pad_ratio=0.15):
    x, y, w, h = box
    pad = int(max(w, h) * pad_ratio)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_shape[1], x + w + pad)
    y2 = min(frame_shape[0], y + h + pad)
    return x1, y1, x2 - x1, y2 - y1


def boxes_overlap(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return False

    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    min_area = min(aw * ah, bw * bh)
    return intersection / float(max(min_area, 1)) > 0.5


def detect_signs(frame):
    red_mask, blue_mask, _, mask = build_color_masks(frame)
    candidates = []

    for color_name, color_mask in (("red", red_mask), ("blue", blue_mask)):
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            scored = score_sign_contour(cnt, frame.shape, color_name)
            if scored is None:
                continue

            score, box = scored
            expanded_box = expand_box(box, frame.shape)
            expanded_box = refine_sign_box_from_color_mask(frame, expanded_box, color_name)
            symbol_score = black_symbol_score(frame, expanded_box)

            if color_name in {"red", "blue"} and symbol_score < 0.015:
                continue

            score += symbol_score * 1200.0

            if any(boxes_overlap(expanded_box, item["box"]) for item in candidates):
                continue

            crop_x, crop_y, crop_w, crop_h = expanded_box
            crop = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            candidates.append(
                {
                    "box": expanded_box,
                    "crop": crop,
                    "score": score,
                    "color": color_name,
                    "symbol_score": symbol_score,
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:MAX_SIGN_CANDIDATES], mask


def annotate_frame(frame, box, name, confidence, secondary_name, secondary_confidence):
    x, y, w, h = box
    color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    label = (
        build_display_label(name, confidence)
        if confidence >= CONFIDENCE_THRESHOLD
        else f"Maybe {build_display_label(name, confidence)}"
    )

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    if confidence < CONFIDENCE_THRESHOLD and secondary_name != name:
        cv2.putText(
            frame,
            f"Next: {build_display_label(secondary_name, secondary_confidence)}",
            (x, min(frame.shape[0] - 10, y + h + 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def save_detection_crops(image_path, detections):
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = Path(image_path).stem

    for index, detection in enumerate(detections, start=1):
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", detection["name"]).strip("_").lower()
        output_path = OUTPUT_DIR / f"{stem}_{index:02d}_{safe_name}.png"
        cv2.imwrite(str(output_path), detection["crop"])


def save_debug_views(image_path, annotated_frame, mask_preview):
    DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)
    stem = Path(image_path).stem
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / f"{stem}_mask_preview.png"), mask_preview)
    cv2.imwrite(str(DEBUG_OUTPUT_DIR / f"{stem}_annotated.png"), annotated_frame)


def iter_image_paths(input_path):
    path = Path(input_path)
    if path.is_dir():
        return sorted(
            [
                file_path
                for file_path in path.iterdir()
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES
            ]
        )
    return [path]


def analyze_frame(frame):
    candidates, mask = detect_signs(frame)
    detections = []

    for candidate in candidates:
        detection = classify_candidate(candidate)
        name = detection["raw_name"]
        conf = detection["raw_confidence"]
        second_name = detection["secondary_name"]
        second_conf = detection["secondary_confidence"]
        speed_value = detection["speed_value"]
        speed_confidence = detection["speed_confidence"]

        if not prediction_is_reliable(
            name,
            conf,
            second_name,
            second_conf,
            speed_value,
            speed_confidence,
            candidate["score"],
            candidate.get("symbol_score", 0.0),
        ):
            continue

        detections.append(detection)

    return detections, mask


def classify_candidate(candidate):
    crop = candidate["crop"]
    name, conf, second_name, second_conf = classify_sign(crop)
    speed_value, speed_confidence = extract_speed_limit_from_crop(crop)
    chosen_name, chosen_conf = refine_prediction_with_speed_value(
        crop,
        name,
        conf,
        second_name,
        second_conf,
        speed_value,
        speed_confidence,
    )

    if speed_value is not None and speed_confidence is not None and speed_confidence >= 70.0:
        chosen_name = f"Speed limit ({speed_value}km/h)"
        chosen_conf = max(chosen_conf, speed_confidence)
        second_name = name
        second_conf = conf

    return {
        "crop": crop,
        "box": candidate["box"],
        "name": chosen_name,
        "confidence": chosen_conf,
        "secondary_name": second_name,
        "secondary_confidence": second_conf,
        "raw_name": name,
        "raw_confidence": conf,
        "speed_value": speed_value,
        "speed_confidence": speed_confidence,
        "candidate_score": candidate["score"],
        "symbol_score": candidate.get("symbol_score", 0.0),
        "color": candidate.get("color"),
    }


def analyze_image(image_path):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detections, mask = analyze_frame(frame)
    output = frame.copy()
    mask_preview, display_mask = build_mask_preview(frame)

    if not detections:
        print(f"No sign detected in {image_path}")
    else:
        print(f"Detected {len(detections)} sign candidate(s) in {image_path}:")
        for index, detection in enumerate(detections, start=1):
            annotate_frame(
                output,
                detection["box"],
                detection["name"],
                detection["confidence"],
                detection["secondary_name"],
                detection["secondary_confidence"],
            )
            print(
                f"  {index}. {build_display_label(detection['name'], detection['confidence'])}"
            )
            cv2.imshow(f"Traffic Sign Crop {index}", detection["crop"])
        save_detection_crops(image_path, detections)
        print(f"Cropped signs saved in: {OUTPUT_DIR.resolve()}")

    save_debug_views(image_path, output, mask_preview)
    print(f"Debug views saved in: {DEBUG_OUTPUT_DIR.resolve()}")

    cv2.imshow("Traffic Sign Mask", display_mask)
    cv2.imshow("Traffic Sign Mask Preview", mask_preview)
    cv2.imshow("Traffic Sign Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "."

    for image_path in iter_image_paths(input_path):
        analyze_image(image_path)


if __name__ == "__main__":
    main()
