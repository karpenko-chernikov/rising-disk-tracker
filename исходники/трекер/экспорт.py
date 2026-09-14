"""Экспорт Excel/CSV и обновление сводки дня."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from трекер.сводка_дня import строка_сводки_с_описаниями

# ключ → (короткий заголовок, подробное описание)
_СЛОВАРЬ: dict[str, tuple[str, str]] = {
    "t_с": ("t, с", "Время от начала ролика, секунды."),
    "Y_м": ("Y, м", "Горизонтальная координата ЦЕНТРА диска (вправо по кадру). Нуль ≈ первые 2 с."),
    "Z_м": ("Z, м", "Вертикальная координата центра (вверх по кадру). Прокси «подъёма»."),
    "Y_сглаж_м": ("Y сглаж., м", "Y после сглаживания Savitzky–Golay (для производных)."),
    "Z_сглаж_м": ("Z сглаж., м", "Z после сглаживания."),
    "beta_рад": ("β, рад", "Угол поворота диска (развёрнутый), из посадки ArUco."),
    "beta_сглаж_рад": ("β сглаж., рад", "Сглаженный угол; розовая стрелка на видео."),
    "Omega_рад_с": ("Ω, рад/с", "Угловая скорость ДИСКА dβ/dt. Не путать с n (колесо привода)."),
    "Omega_dot_рад_с2": ("dΩ/dt, рад/с²", "Угловое ускорение диска."),
    "vY_м_с": ("vY, м/с", "Скорость центра по Y."),
    "vZ_м_с": ("vZ, м/с", "Скорость центра по Z."),
    "aY_м_с2": ("aY, м/с²", "Ускорение центра по Y."),
    "aZ_м_с2": ("aZ, м/с²", "Ускорение центра по Z."),
    "r_cm_м": ("r_ЦМ, м", "Расстояние центра от нуля √(Y²+Z²)."),
    "Ek_rot_Дж": ("Ek вращ., Дж", "Оценка ½ I Ω², I≈½ m R²."),
    "Ek_trans_Дж": ("Ek поступ., Дж", "Оценка ½ m (vY²+vZ²)."),
    "E_grav_Дж": ("E грав., Дж", "Прокси m g Z (нуль относительный)."),
    "качество": ("качество Q", "0…1, согласованность меток с раскладкой при оценке центра."),
    "n_меток": ("число меток", "Сколько ArUco видно в кадре (0–3)."),
    "cx_px": ("cx, px", "Центр диска, пиксель X."),
    "cy_px": ("cy, px", "Центр диска, пиксель Y (вниз по кадру)."),
    "маска_трека": ("есть трек", "1 = центр посчитан в кадре, 0 = нет."),
}


def сохранить_ряд(табл: dict[str, Any], out_dir: Path) -> None:
    df = pd.DataFrame(табл)
    df.to_csv(out_dir / "ряд_по_времени.csv", index=False, float_format="%.8g")
    rename = {k: _СЛОВАРЬ[k][0] for k in df.columns if k in _СЛОВАРЬ}
    df_ru = df.rename(columns=rename)
    словарь = pd.DataFrame(
        [
            {
                "колонка_в_Excel": v[0],
                "ключ": k,
                "что_означает": v[1],
            }
            for k, v in _СЛОВАРЬ.items()
        ]
    )
    with pd.ExcelWriter(out_dir / "ряд_по_времени.xlsx", engine="openpyxl") as w:
        df_ru.to_excel(w, sheet_name="ряд", index=False)
        словарь.to_excel(w, sheet_name="словарь", index=False)


def обновить_сводку(день_dir: Path, строка: dict[str, Any]) -> None:
    path = день_dir / "сводка.xlsx"
    new = pd.DataFrame([строка])
    if path.exists():
        old = pd.read_excel(path)
        if "ролик" in old.columns:
            old = old[old["ролик"] != строка.get("ролик")]
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    desc = строка_сводки_с_описаниями()
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="сводка", index=False)
        pd.DataFrame(
            [{"колонка": k, "что_означает": v} for k, v in desc.items()]
        ).to_excel(w, sheet_name="словарь", index=False)
