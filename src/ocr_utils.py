import cv2
import numpy as np
from difflib import SequenceMatcher

CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

TYPE_TO_LOCATION = {
    "SIZE": 1,
    "VICTIM": 2,
    "CRIME": 3,
    "TIME": 4,
    "PLACE": 5,
    "MOTIVE": 6,
    "WEAPON": 7,
    "BANDIT": 8,
}


def normalize_text(text):
    return "".join(ch for ch in text.upper() if ch.isalnum())


def match_type_text(text, min_score=0.45):
    text = normalize_text(text)
    if not text:
        return ""

    if text in TYPE_TO_LOCATION:
        return text

    best = ""
    best_score = 0.0

    for option in TYPE_TO_LOCATION:
        score = SequenceMatcher(None, text, option).ratio()
        if option.startswith(text) or text.startswith(option):
            score += 0.2
        if score > best_score:
            best = option
            best_score = score

    return best if best_score >= min_score else ""


def type_to_location(type_text):
    matched = match_type_text(type_text)
    return TYPE_TO_LOCATION.get(matched)


def extract_sign_crops(img):
    def order_points(pts):
        s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
        return np.float32([
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)]
        ])

    def approx_quad(cnt):
        for c in (cnt, cv2.convexHull(cnt)):
            p = cv2.arcLength(c, True)
            for e in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05):
                a = cv2.approxPolyDP(c, e * p, True)
                if len(a) == 4:
                    return a.reshape(4, 2).astype(np.float32)
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (100, 120, 40), (140, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    blue_contours = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not blue_contours:
        return [], []

    x, y, w, h = cv2.boundingRect(max(blue_contours, key=cv2.contourArea))

    roi_hsv = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
    gray = cv2.inRange(roi_hsv, (0, 0, 40), (180, 60, 220))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    gray_contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not gray_contours:
        return [], []

    poly = approx_quad(max(gray_contours, key=cv2.contourArea))
    if poly is None:
        return [], []

    poly[:, 0] += x
    poly[:, 1] += y
    src = order_points(poly)

    W = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    H = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])

    zoomed = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H))
    zoomed = cv2.resize(
        cv2.resize(zoomed, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
        (600, 400)
    )

    b, g, r = cv2.split(zoomed)
    clean = cv2.subtract(b, cv2.max(g, r))
    # clean = cv2.GaussianBlur(clean, (3, 3), 0)
    clean = cv2.threshold(clean, 40, 255, cv2.THRESH_BINARY)[1]
    clean = cv2.erode(clean, np.ones((2, 2), np.uint8), iterations=3)

    contours = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw * bh >= 100:
            boxes.append((bx, by, bw, bh))

    Hc, Wc = clean.shape
    type_y = int(0.18 * Hc)
    clue_y = int(0.68 * Hc)
    type_tol = int(0.12 * Hc)
    clue_tol = int(0.15 * Hc)

    type_boxes, clue_boxes = [], []
    for bx, by, bw, bh in boxes:
        cy = by + bh / 2
        if abs(cy - type_y) <= type_tol:
            type_boxes.append((bx, by, bw, bh))
        elif abs(cy - clue_y) <= clue_tol:
            clue_boxes.append((bx, by, bw, bh))

    type_boxes.sort(key=lambda b: b[0])
    clue_boxes.sort(key=lambda b: b[0])

    type_chars, clue_chars = [], []

    for bx, by, bw, bh in type_boxes:
        crop = clean[by:by + bh, bx:bx + bw]
        type_chars.append(cv2.resize(crop, (100, 100), interpolation=cv2.INTER_NEAREST))

    for bx, by, bw, bh in clue_boxes:
        crop = clean[by:by + bh, bx:bx + bw]
        clue_chars.append(cv2.resize(crop, (100, 100), interpolation=cv2.INTER_NEAREST))

    return type_chars, clue_chars


def _prepare_crop_for_model(crop, model):
    x = crop.astype(np.float32) / 255.0

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(f"Expected model input shape like (None,H,W,C), got {input_shape}")

    _, h, w, c = input_shape

    if h is not None and w is not None and (x.shape[0] != h or x.shape[1] != w):
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST)

    if c == 1:
        x = x[..., np.newaxis]
    elif c == 3:
        x = np.stack([x, x, x], axis=-1)
    else:
        raise ValueError(f"Unsupported channel count: {c}")

    return np.expand_dims(x, axis=0)


def predict_char(crop, model):
    x = _prepare_crop_for_model(crop, model)
    pred = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(pred))
    conf = float(pred[idx])
    return CHARACTERS[idx], conf


def read_sign_from_crops(type_crops, clue_crops, model):
    raw_type = "".join(predict_char(crop, model)[0] for crop in type_crops)
    raw_clue = "".join(predict_char(crop, model)[0] for crop in clue_crops)
    return match_type_text(raw_type), normalize_text(raw_clue)


def read_sign(img, model):
    type_crops, clue_crops = extract_sign_crops(img)
    return read_sign_from_crops(type_crops, clue_crops, model)