"""Пакетная обработка видео из входящие/."""

from __future__ import annotations

import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from трекер.имя import разобрать_имя, w_из_n
from трекер.трек import трек_видео, привязать_метры
from трекер.ряды import построить_таблицу
from трекер.метрики import полная_сводка
from трекер.экспорт import сохранить_ряд, обновить_сводку
from трекер.графики import сохранить_графики
from трекер.оверлей import сохранить_оба_оверлея
from трекер.отчёт import построить_отчёт
from трекер.модель import смоделировать, невязка_с_экспериментом
from трекер.сводка_дня import обновить_карту_и_html

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


def _jsonable(obj):
    """Убрать numpy из мета для json."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k not in {"freq", "amp", "пуанкаре", "Y_модель", "Z_модель", "Omega_модель"}}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


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

    дни: set[Path] = set()
    ok_n = err_n = 0
    for path in файлы:
        try:
            print(f"\n=== {path.name} ===")
            день_dir = _обработать_один(path, корень, r_кол, раскладка, обработанные)
            дни.add(день_dir)
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

    for d in дни:
        обновить_карту_и_html(d)

    print(f"\nГотово: успешно {ok_n}, ошибок {err_n}")
    return 0 if err_n == 0 else 2


def _обработать_один(
    path: Path,
    корень: Path,
    r_кол: float,
    раскладка: dict,
    обработанные: Path,
) -> Path:
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

    print("  ряды и метрики…")
    табл = построить_таблицу(
        кадры,
        fps=float(info["fps"]),
        масса_г=раскладка.get("масса_г"),
        R_мм=раскладка.get("R_мм"),
    )
    сводка = полная_сводка(табл, info, кадры=кадры, раскладка=раскладка)

    # модель Maas
    модель_сравнение = None
    модель_ряд = {}
    if (
        w is not None
        and params.delta_мм is not None
        and раскладка.get("R_мм")
        and раскладка.get("масса_г")
    ):
        print("  модель Maas…")
        t_end = float(табл["t_с"][-1]) if len(табл["t_с"]) else 0
        # стартовые из первых валидных
        Y0 = Z0 = 0.0
        Om0 = 1.0
        for i in range(len(табл["t_с"])):
            if np.isfinite(табл["Y_м"][i]):
                Y0 = float(табл["Y_м"][i])
                Z0 = float(табл["Z_м"][i]) if np.isfinite(табл["Z_м"][i]) else 0.0
                Om0 = float(табл["Omega_рад_с"][i]) if np.isfinite(табл["Omega_рад_с"][i]) else 1.0
                break
        модель_ряд = смоделировать(
            W_м_с=w,
            delta_м=params.delta_мм / 1000.0,
            R_м=float(раскладка["R_мм"]) / 1000.0,
            масса_кг=float(раскладка["масса_г"]) / 1000.0,
            t_end=t_end,
            Y0=Y0,
            Z0=Z0,
            Omega0=Om0 if abs(Om0) > 0.05 else 1.0,
        )
        модель_сравнение = невязка_с_экспериментом(табл, модель_ряд)

    сохранить_ряд(табл, out)
    сохранить_графики(
        табл,
        fig_dir,
        заголовок=f"{path.stem} | n={params.n_об_мин} d={params.delta_мм}",
        сводка=сводка,
        модель=модель_сравнение,
    )

    print("  оверлеи…")
    hud = {
        "n_об_мин": params.n_об_мин,
        "delta_мм": params.delta_мм,
        "W_мм_с": None if w is None else w * 1000,
        "Omega": табл["Omega_рад_с"],
    }
    сохранить_оба_оверлея(path, кадры, ov_dir, hud=hud)

    мета = {
        "файл": path.name,
        "дата_съёмки": dt.isoformat(timespec="seconds"),
        "дата_папки": день,
        "пояснения": {
            "n_об_мин": "Об/мин ведущего колеса из имени файла",
            "delta_мм": "Горизонтальный сдвиг контакта δ, мм (+ правее центра, − левее; как на видео)",
            "W_мм_с": "Линейная скорость привода W=2πn/60·r_кол",
            "Omega": "Угловая скорость ДИСКА, рад/с",
            "Y_Z": "Координаты центра диска, м (нуль ≈ первые 2 с)",
        },
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
        "сводка": _jsonable(сводка),
        "модель": _jsonable(модель_сравнение) if модель_сравнение else None,
        "статус": "готово",
    }
    (out / "мета.json").write_text(
        json.dumps(мета, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    (out / "отчёт.txt").write_text(
        построить_отчёт(
            имя_файла=path.name,
            день=день,
            info=info,
            params_n=params.n_об_мин,
            params_d=params.delta_мм,
            W_мм_с=None if w is None else w * 1000,
            раскладка=раскладка,
            r_кол=r_кол,
            сводка=сводка,
            модель_сравнение=модель_сравнение,
        ),
        encoding="utf-8",
    )

    обновить_сводку(
        день_dir,
        {
            "ролик": out.name,
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
            "f_peak_Гц": сводка.get("f_peak_Гц"),
            "lambda_1_1_с": сводка.get("lambda_1_1_с"),
            "T_пред_с": сводка.get("T_пред_с"),
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
    return день_dir
