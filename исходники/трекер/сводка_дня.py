"""Сводка за день: карта режимов и HTML-отчёт."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG = "#0a0a14"
FG = "#c8c8d8"


def обновить_карту_и_html(день_dir: Path) -> None:
    path = день_dir / "сводка.xlsx"
    if not path.exists():
        return
    df = pd.read_excel(path)
    if df.empty:
        return

    # карта режимов n–δ
    if {"n_об_мин", "delta_мм", "Omega_sigma"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        x = df["n_об_мин"].astype(float)
        y = df["delta_мм"].astype(float)
        c = df["Omega_sigma"].astype(float)
        sc = ax.scatter(x, y, c=c, s=80, cmap="plasma", edgecolors="white", linewidths=0.4)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("σ(Ω), рад/с — разброс угловой скорости диска")
        ax.set_xlabel("n, об/мин — скорость ведущего колеса (из имени файла)")
        ax.set_ylabel("δ, мм — смещение оси (из имени файла)")
        ax.set_title("Карта серии за день\nЦвет = насколько «неровно» крутится диск")
        ax.tick_params(colors=FG)
        for sp in ax.spines.values():
            sp.set_color("#333355")
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        fig.savefig(день_dir / "карта_режимов.png", dpi=140, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

    # HTML
    rows = []
    for _, r in df.iterrows():
        ролик = r.get("ролик", "")
        link = f"ролики/{ролик}/отчёт.txt"
        rows.append(
            "<tr>"
            f"<td>{ролик}</td>"
            f"<td>{r.get('n_об_мин','')}</td>"
            f"<td>{r.get('delta_мм','')}</td>"
            f"<td>{r.get('режим','')}</td>"
            f"<td>{r.get('Omega_sigma','')}</td>"
            f"<td>{r.get('доля_трека','')}</td>"
            f"<td><a href='{link}'>отчёт</a></td>"
            "</tr>"
        )
    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Сводка — {день_dir.name}</title>
<style>
body {{ font-family: system-ui, sans-serif; background:#0a0a14; color:#c8c8d8; margin:24px; }}
table {{ border-collapse: collapse; width:100%; }}
th, td {{ border:1px solid #333355; padding:8px; text-align:left; }}
th {{ background:#12122a; }}
a {{ color:#00e5ff; }}
img {{ max-width:640px; margin-top:16px; }}
.note {{ color:#8888aa; max-width:720px; }}
</style></head><body>
<h1>Сводка за {день_dir.name}</h1>
<p class="note">
n — об/мин ведущего колеса; δ — смещение оси, мм; режим — эвристика по σ(Y),σ(Ω);
σ(Ω) — разброс угловой скорости диска; доля трека — кадры с посчитанным центром.
Подробности — в отчёте каждого ролика.
</p>
<table>
<tr><th>Ролик</th><th>n, об/мин</th><th>δ, мм</th><th>Режим</th><th>σ(Ω)</th><th>Доля трека</th><th>Отчёт</th></tr>
{''.join(rows)}
</table>
<p><img src="карта_режимов.png" alt="карта режимов"></p>
</body></html>
"""
    (день_dir / "сводка.html").write_text(html, encoding="utf-8")


def строка_сводки_с_описаниями() -> dict[str, str]:
    """Подписи колонок сводки.xlsx (для листа словарь)."""
    return {
        "ролик": "Имя папки ролика (= имя файла без расширения)",
        "файл": "Исходное имя видео",
        "n_об_мин": "Об/мин ведущего колеса из имени файла",
        "delta_мм": "Смещение оси δ, мм, из имени файла",
        "W_мм_с": "Линейная скорость привода W=2πn/60·r_кол, мм/с",
        "режим": "Эвристический режим динамики диска",
        "доля_трека": "Доля кадров с измеренным центром (0…1)",
        "Omega_среднее": "Средняя угловая скорость ДИСКА, рад/с",
        "Omega_sigma": "СКО угловой скорости диска, рад/с",
        "Y_sigma": "СКО координаты Y центра, м",
        "Z_sigma": "СКО координаты Z центра, м",
        "f_peak_Гц": "Главный пик спектра Ω, Гц",
        "lambda_1_1_с": "Оценка показателя Ляпунова, 1/с",
        "T_пред_с": "Горизонт предсказуемости ≈1/λ₁, с",
        "R_мм": "Радиус диска из раскладки меток",
        "масса_г": "Масса диска из раскладки меток",
        "id_диска": "ID диска из раскладки меток",
    }
