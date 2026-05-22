from collections import Counter, deque

import cv2
import numpy as np

from traffic_detection import annotate_frame, classify_sign

DETECTION_WINDOW = "Traffic Sign Detection"
MAX_CANDIDATES = 6
MIN_CLASSIFICATION_CONFIDENCE = 52.0
RELAXED_CLASSIFICATION_CONFIDENCE = 35.0
STABLE_FRAMES = 5
MIN_STABLE_MATCHES = 3


def box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = (aw * ah) + (bw * bh) - intersection
    return intersection / float(max(union, 1))


def expand_box(box, frame_shape, pad_ratio=0.12):
    x, y, w, h = box
    pad = int(max(w, h) * pad_ratio)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_shape[1], x + w + pad)
    y2 = min(frame_shape[0], y + h + pad)
    return (x1, y1, x2 - x1, y2 - y1)


def infer_color_strength(frame, box):
    x, y, w, h = box
    crop = frame[y : y + h, x : x + w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 60, 50]), np.array([180, 255, 255]))
    blue = cv2.inRange(hsv, np.array([90, 60, 50]), np.array([140, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 80, 255]))

    total = float(max(crop.shape[0] * crop.shape[1], 1))
    red_ratio = (np.count_nonzero(red1) + np.count_nonzero(red2)) / total
    blue_ratio = np.count_nonzero(blue) / total
    white_ratio = np.count_nonzero(white) / total

    return red_ratio, blue_ratio, white_ratio


def build_candidate(frame, box, shape, score):
    box = expand_box(box, frame.shape)
    red_ratio, blue_ratio, white_ratio = infer_color_strength(frame, box)
    x, y, w, h = box
    crop = frame[y : y + h, x : x + w]

    return {
        "box": box,
        "crop": crop,
        "shape": shape,
        "score": score,
        "red_ratio": red_ratio,
        "blue_ratio": blue_ratio,
        "white_ratio": white_ratio,
        "symbol_score": 0.05 if shape == "circle" else 0.03,
        "color": "red" if red_ratio >= blue_ratio else "blue",
    }


def classify_shape_candidate(candidate):
    name, confidence, secondary_name, secondary_confidence = classify_sign(candidate["crop"])
    return {
        "crop": candidate["crop"],
        "box": candidate["box"],
        "name": name,
        "confidence": confidence,
        "secondary_name": secondary_name,
        "secondary_confidence": secondary_confidence,
        "raw_name": name,
        "raw_confidence": confidence,
        "speed_value": None,
        "speed_confidence": None,
        "candidate_score": candidate["score"],
        "symbol_score": candidate.get("symbol_score", 0.0),
        "color": candidate.get("color"),
        "shape": candidate.get("shape"),
    }


def is_sign_like(candidate, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = candidate["box"]
    area_ratio = (w * h) / float(max(frame_h * frame_w, 1))
    aspect = w / float(max(h, 1))

    if not 0.55 <= aspect <= 1.6:
        return False
    if not 0.01 <= area_ratio <= 0.35:
        return False

    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    center_offset = (
        abs(center_x - (frame_w / 2.0)) / max(frame_w / 2.0, 1.0)
        + abs(center_y - (frame_h / 2.0)) / max(frame_h / 2.0, 1.0)
    ) / 2.0
    if center_offset > 0.65:
        return False

    if candidate["shape"] == "circle":
        return (
            candidate["white_ratio"] >= 0.18
            and max(candidate["red_ratio"], candidate["blue_ratio"]) >= 0.03
        )

    if candidate["shape"] == "triangle":
        return candidate["red_ratio"] >= 0.025 and candidate["white_ratio"] >= 0.15

    if candidate["shape"] == "rectangle":
        return candidate["blue_ratio"] >= 0.04 or candidate["white_ratio"] >= 0.35

    return False


def nms_candidates(candidates, iou_threshold=0.35):
    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    kept = []

    for candidate in candidates:
        if all(box_iou(candidate["box"], other["box"]) < iou_threshold for other in kept):
            kept.append(candidate)

    return kept


def detect_shape_candidates(frame):
    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 80, 180)

    candidates = []

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, int(min(frame_h, frame_w) * 0.16)),
        param1=120,
        param2=28,
        minRadius=max(22, int(min(frame_h, frame_w) * 0.05)),
        maxRadius=int(min(frame_h, frame_w) * 0.26),
    )

    if circles is not None:
        for cx, cy, radius in np.round(circles[0]).astype(int):
            x1 = max(0, int(cx - radius * 1.1))
            y1 = max(0, int(cy - radius * 1.1))
            x2 = min(frame_w, int(cx + radius * 1.1))
            y2 = min(frame_h, int(cy + radius * 1.1))
            w = x2 - x1
            h = y2 - y1
            if w < 50 or h < 50:
                continue

            area_ratio = (w * h) / float(max(frame_area, 1))
            if not 0.012 <= area_ratio <= 0.28:
                continue

            center_bonus = 1.0 - (
                (abs(cx - frame_w / 2.0) / max(frame_w / 2.0, 1.0))
                + (abs(cy - frame_h / 2.0) / max(frame_h / 2.0, 1.0))
            ) / 2.0
            candidate = build_candidate(
                frame,
                (x1, y1, w, h),
                "circle",
                (radius * 45.0) + (center_bonus * 1400.0),
            )
            if is_sign_like(candidate, frame.shape):
                candidates.append(candidate)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.003 or area > frame_area * 0.18:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        vertices = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
        if w < 50 or h < 50:
            continue

        shape = None
        if vertices == 3:
            shape = "triangle"
        elif 4 <= vertices <= 6:
            shape = "rectangle"

        if shape is None:
            continue

        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)
        center_bonus = 1.0 - (
            (abs(center_x - frame_w / 2.0) / max(frame_w / 2.0, 1.0))
            + (abs(center_y - frame_h / 2.0) / max(frame_h / 2.0, 1.0))
        ) / 2.0
        candidate = build_candidate(
            frame,
            (x, y, w, h),
            shape,
            area + (center_bonus * 1000.0),
        )
        if is_sign_like(candidate, frame.shape):
            candidates.append(candidate)

    return nms_candidates(candidates)[:MAX_CANDIDATES]


def smooth_detections(history):
    if not history:
        return None

    labels = [item["name"] for item in history]
    label, count = Counter(labels).most_common(1)[0]
    if count < MIN_STABLE_MATCHES:
        return None

    matching = [item for item in history if item["name"] == label]
    matching.sort(key=lambda item: item["confidence"], reverse=True)
    return matching[0]


def choose_realtime_detection(candidates, frame_shape, history):
    detections = [classify_shape_candidate(candidate) for candidate in candidates]
    if not detections:
        history.clear()
        return None

    frame_h, frame_w = frame_shape[:2]
    frame_area = frame_h * frame_w

    scored = []
    for detection, candidate in zip(detections, candidates):
        x, y, w, h = detection["box"]
        area_ratio = (w * h) / float(max(frame_area, 1))
        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)
        center_bonus = 1.0 - (
            (abs(center_x - frame_w / 2.0) / max(frame_w / 2.0, 1.0))
            + (abs(center_y - frame_h / 2.0) / max(frame_h / 2.0, 1.0))
        ) / 2.0
        color_bonus = (
            max(candidate.get("red_ratio", 0.0), candidate.get("blue_ratio", 0.0)) * 180.0
            + candidate.get("white_ratio", 0.0) * 60.0
        )

        score = (
            detection["confidence"]
            + (area_ratio * 2200.0)
            + (center_bonus * 20.0)
            + color_bonus
        )
        scored.append((score, detection))

    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][1]

    if best["confidence"] < MIN_CLASSIFICATION_CONFIDENCE:
        history.clear()
        return None

    history.append(best)
    return smooth_detections(history) or best


def select_best_detection(frame, candidates, min_confidence):
    if not candidates:
        return None

    detections = [classify_shape_candidate(candidate) for candidate in candidates]
    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_h * frame_w
    scored = []

    for detection, candidate in zip(detections, candidates):
        x, y, w, h = detection["box"]
        area_ratio = (w * h) / float(max(frame_area, 1))
        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)
        center_bonus = 1.0 - (
            (abs(center_x - frame_w / 2.0) / max(frame_w / 2.0, 1.0))
            + (abs(center_y - frame_h / 2.0) / max(frame_h / 2.0, 1.0))
        ) / 2.0
        color_bonus = (
            max(candidate.get("red_ratio", 0.0), candidate.get("blue_ratio", 0.0)) * 180.0
            + candidate.get("white_ratio", 0.0) * 60.0
        )

        score = detection["confidence"] + (area_ratio * 2200.0) + (center_bonus * 20.0) + color_bonus
        scored.append((score, detection))

    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][1]
    if best["confidence"] < min_confidence:
        return None
    return best


def analyze_realtime_frame(frame, history=None, min_confidence=MIN_CLASSIFICATION_CONFIDENCE):
    candidates = detect_shape_candidates(frame)
    if history is None:
        detection = select_best_detection(frame, candidates, min_confidence)
        return detection, candidates, None

    detection = choose_realtime_detection(candidates, frame.shape, history)
    return detection, candidates, history


def analyze_single_frame(frame):
    detection, candidates, _ = analyze_realtime_frame(
        frame,
        history=None,
        min_confidence=RELAXED_CLASSIFICATION_CONFIDENCE,
    )
    if detection is not None:
        return detection, candidates

    frame_h, frame_w = frame.shape[:2]
    fallback_size = int(min(frame_h, frame_w) * 0.55)
    x = max(0, (frame_w - fallback_size) // 2)
    y = max(0, (frame_h - fallback_size) // 2)
    fallback_candidate = build_candidate(
        frame,
        (x, y, fallback_size, fallback_size),
        "circle",
        1.0,
    )
    fallback_detection = classify_shape_candidate(fallback_candidate)
    return fallback_detection, candidates


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera 0")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow(DETECTION_WINDOW, cv2.WINDOW_NORMAL)

    print("Press ESC or Q to exit")
    history = deque(maxlen=STABLE_FRAMES)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection, candidates, history = analyze_realtime_frame(frame, history)

        if detection is not None:
            annotate_frame(
                frame,
                detection["box"],
                detection["name"],
                detection["confidence"],
                detection["secondary_name"],
                detection["secondary_confidence"],
            )

        cv2.putText(
            frame,
            f"Candidates: {len(candidates)}  Displayed: {1 if detection is not None else 0}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
        )
        cv2.imshow(DETECTION_WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

        if cv2.getWindowProperty(DETECTION_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
