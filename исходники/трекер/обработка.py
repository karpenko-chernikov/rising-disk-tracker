"""Пакетная обработка видео из входящие/."""

from __future__ import annotations

import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import yaml

from трекер.имя import разобрать_имя, w_из_n
from трекер.трек import трек_видео, привязать_метры
from трекер.ряды import построить_таблицу, сводка_ролика
from трекер.экспорт import сохранить_ряд, обновить_сводку
from трекер.графики import сохранить_графики
from трекер.оверлей import сохранить_оверлей

МЕСЯЦЫ = [
    "",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]


def дата_по_русски(dt: datetime) -> str:
    return f"{dt.day} {МЕСЯЦЫ[dt.month]} {dt.year}"


def дата_съёмки(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)


def запустить_обработку(*, корень: Path, только: str | None = None) -> int:
    входящие = корень / "входящие"
    обработанные = корень / "обработанные"
    ошибки = корень / "ошибки"
    прогоны = корень / "прогоны"
    for d in (входящие, обработанные, ошибки, прогоны):
        d.mkdir(parents=True, exist_ok=True)

    уст = {}
    p_уст = корень / "настройки" / "установка.yaml"
    if p_уст.exists():
        уст = yaml.safe_load(p_уст.read_text(encoding="utf-8")) or {}
    r_кол = float(уст.get("радиус_ведущего_колеса_мм", 15))

    раскладка = {}
    p_р = корень / "калибровка" / "наклейки" / "раскладка_текущая.yaml"
    if p_р.exists():
        раскладка = yaml.safe_load(p_р.read_text(encoding="utf-8")) or {}
    if not раскладка.get("сторона_aruco_мм"):
        print("Нет раскладка_текущая.yaml — сначала: ./трекер.sh метки")
        return 1

    файлы: list[Path] = []
    if только:
        p = Path(только)
        if not p.is_absolute():
            p = входящие / p
        файлы = [p]
    else:
        for p in sorted(входящие.iterdir()):
            if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
                файлы.append(p)

    if not файлы:
        print(f"Нет видео во «{входящие}». Положите файлы вида n120_d-10.mov")
        return 0

    ok_n = err_n = 0
    for path in файлы:
        try:
            print(f"\n=== {path.name} ===")
            _обработать_один(path, корень, r_кол, раскладка, обработанные)
            ok_n += 1
        except Exception as e:
            err_n += 1
            dest = ошибки / path.name
            if path.exists():
                shutil.move(str(path), str(dest))
            (ошибки / f"{path.stem}.ошибка.txt").write_text(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}\n",
                encoding="utf-8",
            )
            print(f"ОШИБКА {path.name}: {e} → ошибки/")

    print(f"\nГотово: успешно {ok_n}, ошибок {err_n}")
    return 0 if err_n == 0 else 2


def _обработать_один(
    path: Path,
    корень: Path,
    r_кол: float,
    раскладка: dict,
    обработанные: Path,
) -> None:
    params = разобрать_имя(path.name)
    dt = дата_съёмки(path)
    день = дата_по_русски(dt)
    день_dir = корень / "прогоны" / день
    out = день_dir / "ролики" / path.stem
    if out.exists():
        out = день_dir / "ролики" / f"{path.stem}_{datetime.now().strftime('%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "графики"
    ov_dir = out / "наложения"
    fig_dir.mkdir(exist_ok=True)
    ov_dir.mkdir(exist_ok=True)

    w = None
    if params.n_об_мин is not None:
        w = w_из_n(params.n_об_мин, r_кол)

    print("  трек ArUco…")
    кадры, info = трек_видео(path, раскладка)
    кадры, origin_meta = привязать_метры(кадры)

    print("  ряды и сводка…")
    табл = построить_таблицу(
        кадры,
        fps=float(info["fps"]),
        масса_г=раскладка.get("масса_г"),
        R_мм=раскладка.get("R_мм"),
    )
    сводка = сводка_ролика(табл, info)

    сохранить_ряд(табл, out)
    сохранить_графики(
        табл,
        fig_dir,
        заголовок=f"{path.stem} | n={params.n_об_мин} d={params.delta_мм}",
    )

    print("  оверлей…")
    hud = {
        "n_об_мин": params.n_об_мин,
        "delta_мм": params.delta_мм,
        "W_мм_с": None if w is None else w * 1000,
    }
    # для оверлея нужны Y_м уже в кадрах — привязать_метры сделал
    # Omega в HUD — из таблицы по индексу; оверлей пишет beta
    сохранить_оверлей(path, кадры, ov_dir / "трек.mp4", hud=hud)

    # превью: первый хороший кадр уже в оверлее; статичный кадр из середины
    мета = {
        "файл": path.name,
        "дата_съёмки": dt.isoformat(timespec="seconds"),
        "дата_папки": день,
        "n_об_мин": params.n_об_мин,
        "delta_мм": params.delta_мм,
        "W_м_с": w,
        "W_мм_с": None if w is None else w * 1000,
        "r_кол_мм": r_кол,
        "раскладка": {
            "id_диска": раскладка.get("id_диска"),
            "R_мм": раскладка.get("R_мм"),
            "масса_г": раскладка.get("масса_г"),
            "сторона_aruco_мм": раскладка.get("сторона_aruco_мм"),
        },
        "видео": info,
        "калибровка_кадра": origin_meta,
        "сводка": сводка,
        "статус": "готово",
    }
    (out / "мета.json").write_text(
        json.dumps(мета, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    отчёт = [
        f"Ролик: {path.name}",
        f"Дата: {день}",
        f"fps: {info.get('fps')}  кадров: {info.get('n_frames')}",
        f"n = {params.n_об_мин} об/мин" if params.n_об_мин is not None else "n: не указано",
        f"δ = {params.delta_мм} мм" if params.delta_мм is not None else "δ: не указано",
        f"W ≈ {w*1000:.2f} мм/с" if w is not None else "W: нет",
        f"Диск: R={раскладка.get('R_мм')} мм, m={раскладка.get('масса_г')} г, id={раскладка.get('id_диска')}",
        "",
        f"Режим (эвристика): {сводка['режим']}",
        f"Доля кадров с треком: {сводка['доля_кадров_с_треком']:.1%}",
        f"Ω среднее: {сводка['Omega_среднее']}",
        f"σ(Ω): {сводка['Omega_sigma']}",
        f"σ(Y): {сводка['Y_sigma']}  σ(Z): {сводка['Z_sigma']}",
        f"f_peak: {сводка['f_peak_Гц']} Гц",
        "",
        "Файлы: ряд_по_времени.xlsx, графики/, наложения/трек.mp4",
        "",
    ]
    if сводка["доля_кадров_с_треком"] < 0.3:
        отчёт.insert(-2, "⚠ Мало кадров с метками — проверьте свет, фокус, размер ArUco, ракурс.")
    (out / "отчёт.txt").write_text("\n".join(отчёт), encoding="utf-8")

    обновить_сводку(
        день_dir,
        {
            "ролик": path.stem,
            "файл": path.name,
            "n_об_мин": params.n_об_мин,
            "delta_мм": params.delta_мм,
            "W_мм_с": None if w is None else w * 1000,
            "режим": сводка["режим"],
            "доля_трека": сводка["доля_кадров_с_треком"],
            "Omega_среднее": сводка["Omega_среднее"],
            "Omega_sigma": сводка["Omega_sigma"],
            "Y_sigma": сводка["Y_sigma"],
            "Z_sigma": сводка["Z_sigma"],
            "f_peak_Гц": сводка["f_peak_Гц"],
            "R_мм": раскладка.get("R_мм"),
            "масса_г": раскладка.get("масса_г"),
            "id_диска": раскладка.get("id_диска"),
        },
    )

    dest = обработанные / path.name
    if dest.exists():
        dest = обработанные / f"{path.stem}_{datetime.now().strftime('%H%M%S')}{path.suffix}"
    shutil.move(str(path), str(dest))
    print(f"OK → прогоны/{день}/ролики/{out.name}/")
