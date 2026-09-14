"""Детекция ArUco и оценка позы диска по кадрам."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class КадрПоза:
    t: float
    cx_px: float | None = None
    cy_px: float | None = None
    Y_м: float | None = None
    Z_м: float | None = None
    beta_рад: float | None = None
    мм_на_px: float | None = None
    n_меток: int = 0
    ids: list[int] = field(default_factory=list)
    quality: float = 0.0
    углы_меток_px: dict[int, np.ndarray] = field(default_factory=dict)  # id -> (4,2)
    центры_меток_px: dict[int, tuple[float, float]] = field(default_factory=dict)


def _словарь(имя: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, имя))


def _детектор(словарь):
    params = cv2.aruco.DetectorParameters()
    # чуть мягче для смаза / малого размера
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.minMarkerPerimeterRate = 0.01
    return cv2.aruco.ArucoDetector(словарь, params)


def _сторона_px(corners: np.ndarray) -> float:
    # corners: (4,2)
    d = [
        np.linalg.norm(corners[0] - corners[1]),
        np.linalg.norm(corners[1] - corners[2]),
        np.linalg.norm(corners[2] - corners[3]),
        np.linalg.norm(corners[3] - corners[0]),
    ]
    return float(np.mean(d))


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P,Q: (N,2) модель и наблюдение в одних единицах (мм).
    Возвращает R (2x2), t (2,) такие что Q ≈ P @ R.T + t.
    """
    assert P.shape == Q.shape and P.shape[0] >= 2
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    P0 = P - pc
    Q0 = Q - qc
    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = qc - pc @ R.T
    return R, t


def _угол_из_R(R: np.ndarray) -> float:
    return float(math.atan2(R[1, 0], R[0, 0]))


def поза_из_меток(
    centers_px: dict[int, tuple[float, float]],
    sides_px: dict[int, float],
    раскладка: dict[str, Any],
) -> tuple[float, float, float, float, float] | None:
    """Возвращает cx_px, cy_px, beta, мм_на_px, quality или None."""
    a_мм = float(раскладка["сторона_aruco_мм"])
    r_m = float(раскладка["радиус_меток_мм"])
    углы = {int(k): float(v) for k, v in раскладка["углы_меток_град"].items()}

    ids = [i for i in centers_px if i in углы]
    if len(ids) < 1:
        return None

    sides = [sides_px[i] for i in ids if i in sides_px]
    if not sides:
        return None
    px_на_мм = float(np.median(sides) / a_мм)
    if px_на_мм <= 1e-6:
        return None
    мм_на_px = 1.0 / px_на_мм

    if len(ids) == 1:
        i = ids[0]
        th = math.radians(углы[i])
        # без ориентации маркера угол диска ненадёжен — берём только центр
        # центр диска ≈ центр метки − r_m * направление (неизвестно без β)
        # используем ориентацию квадрата из corners — передадим отдельно; здесь fallback:
        return None

    # модель в мм: x вправо, y вверх; в кадре y вниз → в Q инвертируем cy
    P = []
    Q = []
    for i in ids:
        th = math.radians(углы[i])
        P.append([r_m * math.cos(th), r_m * math.sin(th)])
        cx, cy = centers_px[i]
        Q.append([cx * мм_на_px, -cy * мм_на_px])
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R, t = _kabsch(P, Q)
    beta = _угол_из_R(R)
    # t — центр в мм (x вправо, y вверх) → обратно в px (y вниз)
    cx_px = t[0] / мм_на_px
    cy_px = -t[1] / мм_на_px
    pred = (P @ R.T) + t
    err = np.linalg.norm(pred - Q, axis=1).mean()
    quality = float(min(1.0, len(ids) / 3.0) * max(0.0, 1.0 - err / max(r_m, 1.0)))
    return cx_px, cy_px, beta, мм_на_px, quality


def поза_одна_метка(
    id_: int,
    corners: np.ndarray,
    раскладка: dict[str, Any],
) -> tuple[float, float, float, float, float] | None:
    """Одна метка: центр + угол из ориентации квадрата."""
    a_мм = float(раскладка["сторона_aruco_мм"])
    r_m = float(раскладка["радиус_меток_мм"])
    углы = {int(k): float(v) for k, v in раскладка["углы_меток_град"].items()}
    if id_ not in углы:
        return None
    side = _сторона_px(corners)
    мм_на_px = a_мм / side
    v = corners[1] - corners[0]
    # угол стороны в системе изображения (y вниз) → переводим в y-вверх
    ang_marker = -math.atan2(v[1], v[0])
    th = math.radians(углы[id_])
    beta = ang_marker - th
    c = corners.mean(axis=0)
    dx = r_m * math.cos(th + beta) / мм_на_px
    dy = -r_m * math.sin(th + beta) / мм_на_px  # в px, y вниз
    cx = float(c[0] - dx)
    cy = float(c[1] - dy)
    return cx, cy, beta, мм_на_px, 0.35


def трек_видео(
    path: Path,
    раскладка: dict[str, Any],
    *,
    макс_кадров: int | None = None,
    прогресс_каждые: int = 60,
) -> tuple[list[КадрПоза], dict[str, Any]]:
    """Проходит видео, возвращает список поз и info (fps, размер…)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    словарь = _словарь(раскладка.get("семейство_aruco", "DICT_4X4_50"))
    det = _детектор(словарь)

    кадры: list[КадрПоза] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if макс_кадров is not None and i >= макс_кадров:
            break
        t = i / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detectMarkers(gray)

        centers: dict[int, tuple[float, float]] = {}
        sides: dict[int, float] = {}
        углы_px: dict[int, np.ndarray] = {}
        if ids is not None and len(ids):
            for c, idv in zip(corners, ids.flatten()):
                id_ = int(idv)
                pts = c.reshape(4, 2).astype(float)
                углы_px[id_] = pts
                centers[id_] = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
                sides[id_] = _сторона_px(pts)

        поза = поза_из_меток(centers, sides, раскладка) if len(centers) >= 2 else None
        if поза is None and len(centers) == 1:
            id0 = next(iter(centers))
            поза = поза_одна_метка(id0, углы_px[id0], раскладка)

        k = КадрПоза(t=t, n_меток=len(centers), ids=sorted(centers), углы_меток_px=углы_px, центры_меток_px=centers)
        if поза is not None:
            cx, cy, beta, мм_на_px, q = поза
            k.cx_px = cx
            k.cy_px = cy
            k.beta_рад = beta
            k.мм_на_px = мм_на_px
            k.quality = q
        кадры.append(k)

        i += 1
        if прогресс_каждые and i % прогресс_каждые == 0:
            print(f"  кадр {i}/{n_frames or '?'}…")

    cap.release()
    info = {
        "fps": fps,
        "n_frames": len(кадры),
        "width": w,
        "height": h,
        "путь": str(path),
    }
    return кадры, info


def привязать_метры(
    кадры: list[КадрПоза],
    *,
    окно_нуля_с: float = 2.0,
) -> tuple[list[КадрПоза], dict[str, Any]]:
    """Y,Z в метрах: начало координат — медиана центра за первые окно_нуля_с.
    Оси: +Y вправо по кадру, +Z вверх по кадру (инверсия py).
    """
    good = [k for k in кадры if k.cx_px is not None and k.мм_на_px is not None]
    if not good:
        return кадры, {"origin": None, "доля_трека": 0.0}

    t0 = good[0].t
    early = [k for k in good if k.t <= t0 + окно_нуля_с]
    if not early:
        early = good[: max(1, len(good) // 20)]

    мм_на_px = float(np.median([k.мм_на_px for k in early]))
    ox = float(np.median([k.cx_px for k in early]))
    oy = float(np.median([k.cy_px for k in early]))

    # развернуть beta непрерывно
    betas = []
    last = None
    for k in кадры:
        if k.beta_рад is None:
            betas.append(None)
            continue
        b = k.beta_рад
        if last is not None:
            d = b - last
            d = (d + math.pi) % (2 * math.pi) - math.pi
            b = last + d
        betas.append(b)
        last = b

    for k, b_unw in zip(кадры, betas):
        if k.cx_px is None or k.мм_на_px is None:
            continue
        # используем стабильный масштаб early
        k.Y_м = (k.cx_px - ox) * мм_на_px / 1000.0
        k.Z_м = -(k.cy_px - oy) * мм_на_px / 1000.0  # вверх
        k.beta_рад = b_unw
        k.мм_на_px = мм_на_px

    доля = len(good) / max(len(кадры), 1)
    meta = {
        "origin_px": [ox, oy],
        "мм_на_px": мм_на_px,
        "доля_трека": доля,
        "окно_нуля_с": окно_нуля_с,
    }
    return кадры, meta
