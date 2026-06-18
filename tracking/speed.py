import math
import numpy as np
import cv2

from tracking.config import PX_PER_METER


def estimate_cam_motion(state, gray, h, w):
    small_w = 320
    scale   = small_w / max(float(w), 1.0)
    small_h = max(1, int(h * scale))
    small   = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    small   = cv2.GaussianBlur(small, (5, 5), 0)
    dx, dy, resp = 0.0, 0.0, 0.0
    if state.prev_gray_small is not None and state.prev_gray_small.shape == small.shape:
        (dx_s, dy_s), r = cv2.phaseCorrelate(state.prev_gray_small, small)
        dx, dy = float(dx_s) / max(scale, 1e-6), float(dy_s) / max(scale, 1e-6)
        resp   = float(r)
        if abs(dx) >= 0.20 * w or abs(dy) >= 0.20 * h:
            dx, dy = 0.0, 0.0
    state.prev_gray_small = small
    return dx, dy, resp


def ema_smooth(state, key, val, alpha=0.4, max_jump=18.0):
    prev = state.speed_ema.get(key)
    if prev is not None and val > prev + max_jump:
        val = prev + max_jump
    out = val if prev is None else alpha * val + (1.0 - alpha) * prev
    state.speed_ema[key] = out
    return out


def compute_speed(state, key, cx, cy, cam_dx, cam_dy, cam_resp, fps, is_ball=False):
    t   = state.frame_count / max(fps, 1.0)
    kmh = 0.0
    if key in state.prev_centers:
        pc     = state.prev_centers[key]
        dt     = max(t - pc['t'], 1.0 / fps)
        obj_dx = (cx - pc['x']) - cam_dx
        obj_dy = (cy - pc['y']) - cam_dy
        kmh    = (math.hypot(obj_dx, obj_dy) / dt / PX_PER_METER) * 3.6
        if cam_resp < 0.08:
            raw = (math.hypot(cx - pc['x'], cy - pc['y']) / dt / PX_PER_METER) * 3.6
            kmh = 0.65 * kmh + 0.35 * raw
    state.prev_centers[key] = {'x': cx, 'y': cy, 't': t}
    alpha    = 0.45 if not is_ball else 0.55
    max_jump = 10.0  if not is_ball else 45.0
    return ema_smooth(state, key, max(0.0, kmh), alpha, max_jump)
