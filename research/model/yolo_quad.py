"""Utility per estrarre un poligono a 4 vertici (quad) da un output YOLO.

Contratto
--------
Input tipico da Ultralytics:
- Contorno: result.masks.xy[i] -> array/lista di punti (N,2) in pixel.
- Fallback rectangle:
  - out.boxes.xyxy[i] (axis-aligned) oppure
  - out.obb.xyxyxyxy[i] (4 punti) se stai usando un OBB model.

Output:
- quad float32 (4,2) ordinato TL,TR,BR,BL.

Strategia
---------
1) Prova a semplificare il contorno con approxPolyDP fino a ottenere 4 vertici.
2) Fallback: convex hull + approx.
3) Se ancora fallisce: ritorna la box/bbox di YOLO (rettangolo) come 4 punti.

Nota: questo file NON fa inferenza; lavora solo sui risultati già prodotti da YOLO.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as (top-left, top-right, bottom-right, bottom-left)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def _try_approx_to_4(contour: np.ndarray) -> Optional[np.ndarray]:
    """Try approxPolyDP with multiple eps values until it yields a convex quad."""
    peri = float(cv2.arcLength(contour, True))
    if peri <= 0:
        return None

    # small -> larger epsilon
    for frac in (0.0015, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(contour, frac * peri, True)
        if approx is None:
            continue
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def quad_from_contour_xy(contour_xy: np.ndarray) -> Optional[np.ndarray]:
    """Deriva un quad da un contorno YOLO (Nx2 px)."""
    pts = np.asarray(contour_xy, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return None

    contour = pts.reshape((-1, 1, 2)).astype(np.float32)

    approx = _try_approx_to_4(contour)
    if approx is not None:
        return order_points_tl_tr_br_bl(approx)

    hull = cv2.convexHull(contour)
    approx = _try_approx_to_4(hull)
    if approx is not None:
        return order_points_tl_tr_br_bl(approx)

    return None


def quad_from_xyxy(xyxy: np.ndarray) -> Optional[np.ndarray]:
    """Convert axis-aligned bbox xyxy -> quad (4x2)."""
    arr = np.asarray(xyxy, dtype=np.float32).reshape(-1)
    if arr.size != 4:
        return None
    x1, y1, x2, y2 = arr.tolist()
    quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return order_points_tl_tr_br_bl(quad)


def quad_from_yolo_result(
    result,
    *,
    instance_index: int = 0,
    min_area_px: float = 1000.0,
) -> Optional[np.ndarray]:
    """Estrae un quad da un singolo `ultralytics.engine.results.Results`.

    Tenta:
    1) contour -> quad
    2) OBB corners (se presenti)
    3) bbox xyxy -> quad

    Ritorna None se non trova niente di sensato.
    """

    # 1) Contour from masks.xy
    try:
        masks = getattr(result, "masks", None)
        if masks is not None and getattr(masks, "xy", None) is not None:
            contours = masks.xy
            if 0 <= instance_index < len(contours):
                q = quad_from_contour_xy(np.asarray(contours[instance_index], dtype=np.float32))
                if q is not None:
                    area = abs(cv2.contourArea(q.reshape(-1, 1, 2).astype(np.float32)))
                    if area >= float(min_area_px):
                        return q
    except Exception:
        pass

    # 2) Standard bbox fallback
    try:
        boxes = getattr(result, "boxes", None)
        if boxes is not None and hasattr(boxes, "xyxy"):
            xyxy = boxes.xyxy
            if xyxy is not None and len(xyxy) > instance_index:
                arr = xyxy[instance_index].detach().cpu().numpy()
                q = quad_from_xyxy(arr)
                if q is None:
                    return None
                area = abs(cv2.contourArea(q.reshape(-1, 1, 2).astype(np.float32)))
                if area >= float(min_area_px):
                    return q
                return q
    except Exception:
        pass

    return None

