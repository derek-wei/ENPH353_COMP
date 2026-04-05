import cv2, numpy as np
from tensorflow.keras.models import load_model

MODEL = load_model('/models/353_ocr_model.keras')
CHARS = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))

# weiweiOCR.read_sign(img)
# type_text, clue_text = weiweiOCR.read_sign(img)
# returns type_text, clue_text

def read_sign(img, model=MODEL):
    def order(pts):
        s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
        return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])

    def quad(cnt):
        for c in (cnt, cv2.convexHull(cnt)):
            p = cv2.arcLength(c, True)
            for e in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05):
                a = cv2.approxPolyDP(c, e * p, True)
                if len(a) == 4:
                    return a.reshape(4, 2).astype(np.float32)

    def ocr(chars):
        if not chars:
            return ""
        batch = np.stack([
            (cv2.resize(c if c.ndim == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY),
                        (100, 100), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0)[..., None]
            for c in chars
        ])
        return "".join(CHARS[np.argmax(model.predict(batch, verbose=0), axis=1)])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.morphologyEx(
        cv2.inRange(hsv, (100, 120, 40), (140, 255, 255)),
        cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    cnts = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        return "", ""

    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    roi_hsv = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
    gray = cv2.inRange(roi_hsv, (0, 0, 40), (180, 60, 220))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        return "", ""

    poly = quad(max(cnts, key=cv2.contourArea))
    if poly is None:
        return "", ""

    poly += [x, y]
    src = order(poly)
    W = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    H = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    zoomed = cv2.warpPerspective(
        img, cv2.getPerspectiveTransform(src, np.float32([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]])), (W, H)
    )
    zoomed = cv2.resize(cv2.resize(zoomed, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC), (600, 400))

    b, g, r = cv2.split(zoomed)
    clean = cv2.erode(
        cv2.threshold(cv2.subtract(b, cv2.max(g, r)), 40, 255, cv2.THRESH_BINARY)[1],
        np.ones((2, 2), np.uint8), iterations=3
    )

    boxes = []
    for c in cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw * bh >= 100:
            boxes.append((bx, by, bw, bh))

    Hc = clean.shape[0]
    type_y, clue_y = int(0.18 * Hc), int(0.68 * Hc)
    type_tol, clue_tol = int(0.12 * Hc), int(0.15 * Hc)

    type_boxes, clue_boxes = [], []
    for bx, by, bw, bh in boxes:
        cy = by + bh / 2
        if abs(cy - type_y) <= type_tol:
            type_boxes.append((bx, by, bw, bh))
        elif abs(cy - clue_y) <= clue_tol:
            clue_boxes.append((bx, by, bw, bh))

    type_imgs = [cv2.resize(clean[y:y+h, x:x+w], (100, 100), interpolation=cv2.INTER_NEAREST)
                 for x, y, w, h in sorted(type_boxes, key=lambda b: b[0])]
    clue_imgs = [cv2.resize(clean[y:y+h, x:x+w], (100, 100), interpolation=cv2.INTER_NEAREST)
                 for x, y, w, h in sorted(clue_boxes, key=lambda b: b[0])]

    return ocr(type_imgs), ocr(clue_imgs)