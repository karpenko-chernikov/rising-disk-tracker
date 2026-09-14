"""Генерация PDF меток ArUco и раскладки yaml."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image


def сторона_aruco_мм(r_мм: float, cfg: dict[str, Any]) -> float:
    a = cfg["сторона_aruco_доля_R"] * r_мм
    a = max(cfg["сторона_aruco_мин_мм"], min(cfg["сторона_aruco_макс_мм"], a))
    return round(a, 1)


def загрузить_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def спросить(prompt: str, default: str | None = None) -> str:
    хв = f" [{default}]" if default is not None else ""
    s = input(f"{prompt}{хв}: ").strip()
    if not s and default is not None:
        return default
    return s


def словарь_aruco(имя: str):
    код = getattr(cv2.aruco, имя)
    return cv2.aruco.getPredefinedDictionary(код)


def сгенерировать_маркер_png(словарь, id_: int, px: int = 400) -> Image.Image:
    img = cv2.aruco.generateImageMarker(словарь, id_, px)
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = img
    return Image.fromarray(rgb)


def нарисовать_схему_клейки(
    r_мм: float,
    a_мм: float,
    r_m: float,
    углы: dict[int, float],
    запрет: float,
) -> Image.Image:
    """Схема диск + позиции меток (для второй страницы PDF)."""
    size = 900
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    cx = cy = size // 2
    scale = (size * 0.42) / r_мм  # px / mm

    def to_px(x_мм: float, y_мм: float) -> tuple[int, int]:
        # Y вправо, Z вверх на схеме (как на бумаге)
        return int(cx + x_мм * scale), int(cy - y_мм * scale)

    # диск
    cv2.circle(img, (cx, cy), int(r_мм * scale), (40, 40, 40), 3)
    # запретная зона
    cv2.circle(img, (cx, cy), int(запрет * scale), (180, 180, 255), 2)
    cv2.putText(
        img,
        "zona rolika — pustoe",
        (cx - 120, cy + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (120, 120, 200),
        1,
        cv2.LINE_AA,
    )
    # окружность меток
    cv2.circle(img, (cx, cy), int(r_m * scale), (200, 200, 200), 1)

    for id_, ang in углы.items():
        rad = math.radians(float(ang))
        x = r_m * math.cos(rad)
        y = r_m * math.sin(rad)
        px, py = to_px(x, y)
        half = int(a_мм * scale / 2)
        cv2.rectangle(img, (px - half, py - half), (px + half, py + half), (0, 0, 0), 2)
        cv2.putText(
            img,
            f"ID={id_}",
            (px - half, py - half - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(img, (px, py), 3, (0, 0, 255), -1)

    cv2.putText(
        img,
        f"R={r_мм:.0f} mm  a={a_мм:.1f} mm  r_m={r_m:.1f} mm",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Kamera SPRAVA — kleit na etu storonu diska",
        (20, size - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 100, 0),
        2,
        cv2.LINE_AA,
    )
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def сохранить_pdf(
    путь: Path,
    маркеры: dict[int, Image.Image],
    a_мм: float,
    схема: Image.Image,
    текст_шапки: list[str],
) -> None:
    c = canvas.Canvas(str(путь), pagesize=A4)
    W, H = A4

    # --- страница 1: три квадрата 1:1 ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, H - 20 * mm, "Metki ArUco — pechat masshtab 100%")
    c.setFont("Helvetica", 10)
    y = H - 28 * mm
    for line in текст_шапки:
        c.drawString(20 * mm, y, line)
        y -= 5 * mm

    c.setFont("Helvetica", 9)
    c.drawString(
        20 * mm,
        y - 2 * mm,
        "Proverte lineikoy: storona kvadrata = a mm. Ne «vpisat v list».",
    )

    # три маркера в ряд
    gap = 15 * mm
    x0 = 20 * mm
    y0 = H - 95 * mm
    for i, (id_, im) in enumerate(sorted(маркеры.items())):
        x = x0 + i * (a_мм * mm + gap)
        c.drawImage(
            ImageReader(im),
            x,
            y0,
            width=a_мм * mm,
            height=a_мм * mm,
            preserveAspectRatio=True,
            mask="auto",
        )
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y0 - 6 * mm, f"ID = {id_}")
        # рамка обрезки
        c.rect(x, y0, a_мм * mm, a_мм * mm, stroke=1, fill=0)

    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, 25 * mm, "Vyrezat po ramke, nakleit po sheme (stranica 2).")
    c.showPage()

    # --- страница 2: схема ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, H - 20 * mm, "Shema: kuda kleit (kamera sprava)")
    c.setFont("Helvetica", 10)
    c.drawString(
        20 * mm,
        H - 28 * mm,
        "Centry metok na okruzhnosti r_m. Zona u rolika — pustaya. Tochnost ±2–3 mm OK.",
    )
    max_w = W - 40 * mm
    max_h = H - 50 * mm
    c.drawImage(
        ImageReader(схема),
        20 * mm,
        20 * mm,
        width=max_w,
        height=max_h,
        preserveAspectRatio=True,
        anchor="c",
    )
    c.save()


def запустить_метки(
    *,
    r_мм: float | None,
    диаметр_мм: float | None,
    масса_г: float | None,
    id_диска: str | None,
    интерактив: bool,
    корень: Path,
) -> int:
    cfg = загрузить_yaml(корень / "настройки" / "по_умолчанию.yaml")
    out_dir = корень / "калибровка" / "наклейки"
    out_dir.mkdir(parents=True, exist_ok=True)

    if диаметр_мм is not None and r_мм is None:
        r_мм = диаметр_мм / 2.0

    if интерактив:
        if r_мм is None:
            s = спросить("Радиус диска R, мм (или Enter и введите диаметр отдельно)", "50")
            r_мм = float(s.replace(",", "."))
        if масса_г is None:
            масса_г = float(спросить("Масса диска, г", "50").replace(",", "."))
        if not id_диска:
            авто = f"диск-{datetime.now().strftime('%Y%m%d-%H%M')}"
            id_диска = спросить("ID диска", авто)
        заметка = спросить("Заметка (Enter = пусто)", "")
    else:
        if r_мм is None:
            raise SystemExit("Нужен --R-мм или --диаметр-мм")
        if масса_г is None:
            raise SystemExit("Нужен --масса-г")
        if not id_диска:
            id_диска = f"диск-{datetime.now().strftime('%Y%m%d-%H%M')}"
        заметка = ""

    assert r_мм is not None and масса_г is not None and id_диска

    a_мм = сторона_aruco_мм(r_мм, cfg)
    r_m = round(cfg["доля_радиуса_меток"] * r_мм, 2)
    запрет = round(cfg["доля_запретной_зоны"] * r_мм, 2)
    углы = {int(k): float(v) for k, v in cfg["углы_меток_град"].items()}

    словарь = словарь_aruco(cfg["семейство_aruco"])
    маркеры = {i: сгенерировать_маркер_png(словарь, i) for i in углы}

    раскладка = {
        "версия": 1,
        "id_диска": id_диска,
        "R_мм": float(r_мм),
        "диаметр_мм": float(2 * r_мм),
        "масса_г": float(масса_г),
        "заметка": заметка,
        "семейство_aruco": cfg["семейство_aruco"],
        "сторона_aruco_мм": a_мм,
        "радиус_меток_мм": r_m,
        "запретная_зона_мм": запрет,
        "углы_меток_град": углы,
        "создано": datetime.now().isoformat(timespec="seconds"),
    }

    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in id_диска)
    yaml_path = out_dir / f"раскладка_{safe_id}.yaml"
    pdf_path = out_dir / f"метки_{safe_id}_R{int(round(r_мм))}мм.pdf"

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(раскладка, f, allow_unicode=True, sort_keys=False)

    схема = нарисовать_схему_клейки(r_мм, a_мм, r_m, углы, запрет)
    шапка = [
        f"ID: {id_диска}   R = {r_мм:.1f} mm   m = {масса_г:.1f} g   a = {a_мм:.1f} mm",
        f"r_m = {r_m:.1f} mm   family = {cfg['семейство_aruco']}",
    ]
    if заметка:
        шапка.append(f"note: {заметка}")

    сохранить_pdf(pdf_path, маркеры, a_мм, схема, шапка)

    # актуальная раскладка по умолчанию для обработки
    текущая = out_dir / "раскладка_текущая.yaml"
    with текущая.open("w", encoding="utf-8") as f:
        yaml.safe_dump(раскладка, f, allow_unicode=True, sort_keys=False)

    print()
    print("Готово.")
    print(f"  PDF:  {pdf_path}")
    print(f"  YAML: {yaml_path}")
    print(f"  Также: {текущая}")
    print()
    print("1) Печать 100% (не вписывать в страницу).")
    print(f"2) Линейкой: сторона квадрата = {a_мм} мм.")
    print("3) Вырезать ID 0,1,2 и наклеить по схеме (стр. 2), вне зоны ролика.")
    return 0
