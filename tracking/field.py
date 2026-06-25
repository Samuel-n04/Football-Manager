import numpy as np
import cv2

from tracking.config import FIELD_MASK_REFRESH


def _adaptive_field_range(frame):
    """
    Sample the center-bottom of the frame (always field in football broadcasts)
    and return an HSV range that covers the dominant grass color.
    Works for natural grass, dark synthetic turf, pale synthetic turf, etc.
    """
    h, w = frame.shape[:2]
    # Bottom 35%, center 40%
    y1, y2 = int(h * 0.65), h
    x1, x2 = int(w * 0.30), int(w * 0.70)
    patch = frame[y1:y2, x1:x2]
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

    # Dominant color via k-means (k=1 is just the mean, k=2 handles line markings)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(hsv_patch, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    # Pick the cluster with more pixels
    counts = np.bincount(labels.flatten())
    field_center = centers[np.argmax(counts)]

    h_c, s_c, v_c = field_center
    # Build tolerant range
    h_tol = 25
    s_tol = max(30, s_c * 0.8 + 15)   # generous on saturation (handles desaturated turf)
    v_tol = max(50, v_c * 0.5)

    lo = np.array([max(0,   h_c - h_tol),
                   max(0,   s_c - s_tol),
                   max(15,  v_c - v_tol)], dtype=np.uint8)
    hi = np.array([min(180, h_c + h_tol),
                   255,
                   min(255, v_c + v_tol)], dtype=np.uint8)
    return lo, hi


def detect_field(state, frame):
    if (state.field_mask_cache is not None
            and state.frame_count - state.field_mask_frame < FIELD_MASK_REFRESH):
        return state.field_mask_cache

    h, w = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Adaptive grass color range
    lo, hi = _adaptive_field_range(frame)
    green = cv2.inRange(hsv, lo, hi)

    # Fallback: also always include standard broadcast green
    std_green = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([98, 255, 255]))
    green = cv2.bitwise_or(green, std_green)

    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9),  np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN,  np.ones((5, 5),  np.uint8))

    white          = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 50, 255]))
    green_expanded = cv2.dilate(green, np.ones((40, 40), np.uint8))
    white_on_field = cv2.bitwise_and(white, green_expanded)

    combined = cv2.bitwise_or(green, white_on_field)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    else:
        mask = cv2.dilate(green, np.ones((25, 25), np.uint8))

    state.field_mask_cache = mask
    state.field_mask_frame = state.frame_count
    return mask


def on_field(xyxy, mask, h, w):
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    cx   = max(0, min(w - 1, int((x1 + x2) / 2)))
    foot = min(h - 1, int(y2))
    py1  = max(0, foot - 5);  py2 = min(h - 1, foot + 8)
    px1  = max(0, cx   - 8);  px2 = min(w - 1, cx   + 8)
    patch = mask[py1:py2, px1:px2]
    return float((patch > 0).mean()) > 0.3 or mask[foot, cx] > 0
