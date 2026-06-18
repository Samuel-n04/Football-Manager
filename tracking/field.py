import numpy as np
import cv2

from tracking.config import FIELD_MASK_REFRESH


def detect_field(state, frame):
    if (state.field_mask_cache is not None
            and state.frame_count - state.field_mask_frame < FIELD_MASK_REFRESH):
        return state.field_mask_cache

    h, w = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([98, 255, 255]))
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
