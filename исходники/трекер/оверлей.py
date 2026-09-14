"""Видео с наложением трека."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from трекер.трек import КадрПоза

РежимХвоста = Literal["ускользающий", "непрерывный"]


def сохранить_оверлей(
    video_path: Path,
    кадры: list[КадрПоза],
    out_path: Path,
    *,
    hud: dict[str, Any] | None = None,
    режим_хвоста: РежимХвоста = "ускользающий",
    trail_len: int = 90,
) -> Path:
    """Рисует оверлей.

    Хвост — траектория **центра диска** (не отдельной метки):
    центр считается по всем видимым ArUco (≥2) жёсткой посадкой (Kabsch).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не открыть видео для оверлея: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
            cx, cy = int(round(k.cx_px)), int(round(k.cy_px))
            cv2.circle(frame, (cx, cy), 6, (0, 230, 255), -1)
            trail.append((cx, cy))
            if режим_хвоста == "ускользающий" and len(trail) > trail_len:
                trail = trail[-trail_len:]

            n = len(trail)
            for j in range(1, n):
                if режим_хвоста == "ускользающий":
                    # ярче к «сейчас», тусклее к прошлому
                    f = j / max(n - 1, 1)
                    color = (
                        int(80 + 175 * f),   # B
                        int(60 + 120 * f),   # G
                        0,                  # R → голубой
                    )
                    thickness = 1 if f < 0.4 else 2
                else:
                    color = (255, 180, 0)
                    thickness = 2
                cv2.line(frame, trail[j - 1], trail[j], color, thickness, cv2.LINE_AA)

            if k.beta_рад is not None:
                L = 40
                dx = int(L * np.cos(k.beta_рад))
                dy = int(-L * np.sin(k.beta_рад))
                cv2.arrowedLine(
                    frame, (cx, cy), (cx + dx, cy + dy), (255, 0, 255), 2, tipLength=0.3
                )

        lines = [
            f"t={k.t:.2f}s | markers={k.n_меток}/3 | Q={k.quality:.2f}",
        ]
        if k.Y_м is not None and k.Z_м is not None:
            lines.append(f"Y={k.Y_м * 1000:.1f}mm Z={k.Z_м * 1000:.1f}mm (center)")
        if k.beta_рад is not None:
            lines.append(f"beta={k.beta_рад:.2f}rad (pink arrow)")
        if hud.get("Omega") is not None and i < len(hud["Omega"]):
            om = hud["Omega"][i]
            if om == om:  # not NaN
                lines.append(f"Omega={om:.2f}rad/s (disk spin)")
        if hud.get("n_об_мин") is not None:
            lines.append(f"n={hud['n_об_мин']}rpm (drive wheel) d={hud.get('delta_мм')}mm")
        if hud.get("W_мм_с") is not None:
            lines.append(f"W={hud['W_мм_с']:.1f}mm/s (drive linear)")
        # компактная легенда раз в начале
        if i == 0:
            lines.append("trail=center path | boxes=ArUco")
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
    return out_path


def сохранить_оба_оверлея(
    video_path: Path,
    кадры: list[КадрПоза],
    ov_dir: Path,
    *,
    hud: dict[str, Any] | None = None,
) -> list[Path]:
    """Пишет два ролика: ускользающий хвост и полная траектория."""
    paths = []
    paths.append(
        сохранить_оверлей(
            video_path,
            кадры,
            ov_dir / "трек_хвост.mp4",
            hud=hud,
            режим_хвоста="ускользающий",
        )
    )
    # второй проход — снова читает исходник
    paths.append(
        сохранить_оверлей(
            video_path,
            кадры,
            ov_dir / "трек_полный.mp4",
            hud=hud,
            режим_хвоста="непрерывный",
        )
    )
    return paths
