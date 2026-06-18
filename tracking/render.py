import cv2

from tracking.config import LABEL_SCALE, LABEL_THICK


def draw_label(frame, xyxy, label, color, box_thick=2):
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_THICK)
    ty = max(y1 - 3, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 2), (x1 + tw + 4, ty + 2), color, -1)
    lum     = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    txt_col = (0, 0, 0) if lum > 140 else (255, 255, 255)
    cv2.putText(frame, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_SCALE, txt_col, LABEL_THICK, cv2.LINE_AA)


def slice_detect(mdl, frame, slice_sz=640, overlap=0.25, conf=0.06):
    h, w  = frame.shape[:2]
    step  = int(slice_sz * (1.0 - overlap))
    boxes, confs, cls_ids = [], [], []
    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = frame[y:y + slice_sz, x:x + slice_sz]
            res  = mdl.predict(tile, conf=conf, classes=[0, 32], verbose=False)
            for r in res:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    boxes.append([x + bx1, y + by1, x + bx2, y + by2])
                    confs.append(float(b.conf[0]))
                    cls_ids.append(int(b.cls[0]))
    return boxes, confs, cls_ids


def iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-6)
