"""Синтетическое видео с 3 ArUco для проверки пайплайна."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "исходники"))


def main() -> None:
    раскладка_path = ROOT / "калибровка" / "наклейки" / "раскладка_текущая.yaml"
    if not раскладка_path.exists():
        raise SystemExit("Сначала: ./трекер.sh метки --R-мм 50 --масса-г 50 --id тест --no-интерактив")
    раскладка = yaml.safe_load(раскладка_path.read_text(encoding="utf-8"))

    a_мм = float(раскладка["сторона_aruco_мм"])
    r_m = float(раскладка["радиус_меток_мм"])
    углы = {int(k): float(v) for k, v in раскладка["углы_меток_град"].items()}
    словарь = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, раскладка["семейство_aruco"]))

    # геометрия кадра: 1 px = 0.2 мм → диск R=50 мм = 250 px radius
    мм_на_px = 0.2
    px_на_мм = 1.0 / мм_на_px
    W, H = 960, 720
    fps = 30
    seconds = 4
    n = fps * seconds

    markers = {}
    side_px = max(20, int(round(a_мм * px_на_мм)))
    for id_ in углы:
        markers[id_] = cv2.aruco.generateImageMarker(словарь, id_, side_px)

    out = ROOT / "входящие" / "n100_d-5_synth.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    cx0, cy0 = W / 2, H / 2
    for i in range(n):
        t = i / fps
        # диск едет и крутится
        beta = 1.2 * t
        ox = cx0 + 40 * math.sin(0.7 * t)
        oy = cy0 - 30 * math.cos(0.5 * t)
        frame = np.full((H, W, 3), 30, dtype=np.uint8)
        # белый диск
        R_px = int(раскладка["R_мм"] * px_на_мм)
        cv2.circle(frame, (int(ox), int(oy)), R_px, (240, 240, 240), -1)

        for id_, ang0 in углы.items():
            # в модели +y вверх; в кадре +y вниз
            th = math.radians(ang0) + beta
            mx = ox + r_m * px_на_мм * math.cos(th)
            my = oy - r_m * px_на_мм * math.sin(th)
            M = markers[id_]
            # узор не вращаем (иначе OpenCV часто теряет ID на синтетике);
            # на реальных наклейках маркеры крутятся вместе с диском — там печать чёткая
            Mr = M if M.ndim == 3 else cv2.cvtColor(M, cv2.COLOR_GRAY2BGR)
            x0 = int(round(mx - side_px / 2))
            y0 = int(round(my - side_px / 2))
            x1, y1 = x0 + side_px, y0 + side_px
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
                continue
            roi = frame[y0:y1, x0:x1]
            mask = Mr[:, :, 0] < 200
            roi[mask] = (0, 0, 0)
            roi[~mask] = (255, 255, 255)
            frame[y0:y1, x0:x1] = roi

        writer.write(frame)

    writer.release()
    print(f"Синтетика: {out}")


if __name__ == "__main__":
    main()
