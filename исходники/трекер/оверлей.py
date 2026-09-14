"""Видео с наложением трека."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from трекер.трек import КадрПоза


def сохранить_оверлей(
    video_path: Path,
    кадры: list[КадрПоза],
    out_path: Path,
    *,
    hud: dict[str, Any] | None = None,
    trail_len: int = 90,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не открыть видео для оверлея: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # mp4v широко читается; при проблемах — avi
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_path = out_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    trail: list[tuple[int, int]] = []
    i = 0
    hud = hud or {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= len(кадры):
            break
        k = кадры[i]

        # маркеры
        for id_, pts in k.углы_меток_px.items():
            pts_i = pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts_i], True, (0, 255, 255), 2)
            c = k.центры_меток_px.get(id_)
            if c:
                cv2.putText(
                    frame,
                    f"ID{id_}",
                    (int(c[0]) + 6, int(c[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if k.cx_px is not None and k.cy_px is not None:
            cx, cy = int(k.cx_px), int(k.cy_px)
            cv2.circle(frame, (cx, cy), 6, (0, 230, 255), -1)
            trail.append((cx, cy))
            if len(trail) > trail_len:
                trail = trail[-trail_len:]
            for j in range(1, len(trail)):
                a = int(80 + 175 * j / len(trail))
                cv2.line(frame, trail[j - 1], trail[j], (255, 180, 0), 2)
            # ось тела
            if k.beta_рад is not None and k.мм_на_px:
                L = 40
                dx = int(L * np.cos(k.beta_рад))
                dy = int(-L * np.sin(k.beta_рад))  # y вниз на кадре
                cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (255, 0, 255), 2, tipLength=0.3)

        # HUD
        lines = [
            f"t={k.t:.2f}s  markers={k.n_меток}  Q={k.quality:.2f}",
        ]
        if k.Y_м is not None:
            lines.append(f"Y={k.Y_м*1000:.1f}mm  Z={k.Z_м*1000:.1f}mm")
        if k.beta_рад is not None:
            lines.append(f"beta={k.beta_рад:.2f}rad")
        if hud.get("n_об_мин") is not None:
            lines.append(f"n={hud['n_об_мин']} rpm  d={hud.get('delta_мм')}mm")
        if hud.get("W_мм_с") is not None:
            lines.append(f"W={hud['W_мм_с']:.1f}mm/s")
        y0 = 28
        for line in lines:
            cv2.putText(frame, line, (16, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (16, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
            y0 += 24
        if k.quality < 0.25 or k.n_меток < 2:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)

        writer.write(frame)
        i += 1

    cap.release()
    writer.release()
