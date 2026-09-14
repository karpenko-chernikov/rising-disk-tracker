"""Расширенные научные метрики с человекочитаемыми описаниями."""

from __future__ import annotations

from typing import Any

import numpy as np

from трекер.ряды import спектр_omega, классифицировать


def сечение_пуанкаре(
    t: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    Omega: np.ndarray,
) -> dict[str, np.ndarray]:
    """Пересечения Y=0 снизу вверх → точки (Ω, Z) и карта возврата Ω."""
    ok = np.isfinite(Y) & np.isfinite(Z) & np.isfinite(Omega)
    if ok.sum() < 20:
        return {"Omega": np.array([]), "Z": np.array([]), "Omega_next": np.array([])}
    y, z, o = Y.copy(), Z.copy(), Omega.copy()
    y[~ok] = np.nan
    Omegas, Zs = [], []
    for i in range(1, len(y)):
        if not (np.isfinite(y[i - 1]) and np.isfinite(y[i])):
            continue
        if y[i - 1] < 0 <= y[i]:
            # линейная интерполяция момента пересечения
            w = -y[i - 1] / (y[i] - y[i - 1] + 1e-15)
            Omegas.append((1 - w) * o[i - 1] + w * o[i])
            Zs.append((1 - w) * z[i - 1] + w * z[i])
    Omegas = np.asarray(Omegas, dtype=float)
    Zs = np.asarray(Zs, dtype=float)
    if len(Omegas) < 2:
        return {"Omega": Omegas, "Z": Zs, "Omega_next": np.array([])}
    return {
        "Omega": Omegas[:-1],
        "Z": Zs[:-1],
        "Omega_next": Omegas[1:],
    }


def спектр_метрики(t: np.ndarray, omega: np.ndarray) -> dict[str, Any]:
    freq, amp = спектр_omega(t, omega)
    if len(amp) < 3:
        return {
            "f_peak_Гц": None,
            "ширина_пика_Гц": None,
            "доля_энергии_в_пике": None,
            "описание": "Спектр не посчитан: мало данных по Ω(t).",
        }
    # игнор нулевой частоты
    amp2 = amp.copy()
    amp2[0] = 0
    i = int(np.argmax(amp2))
    f_peak = float(freq[i])
    # ширина на половине высоты
    half = amp2[i] / 2
    left = i
    while left > 1 and amp2[left] >= half:
        left -= 1
    right = i
    while right < len(amp2) - 1 and amp2[right] >= half:
        right += 1
    width = float(freq[min(right, len(freq) - 1)] - freq[max(left, 0)])
    total = float(np.sum(amp2**2) + 1e-15)
    # энергия в окне ±width вокруг пика
    mask = (freq >= f_peak - width) & (freq <= f_peak + width)
    frac = float(np.sum(amp2[mask] ** 2) / total)
    return {
        "f_peak_Гц": f_peak,
        "ширина_пика_Гц": width,
        "доля_энергии_в_пике": frac,
        "freq": freq,
        "amp": amp,
        "описание": (
            f"Главная частота колебаний Ω ≈ {f_peak:.3f} Гц; "
            f"ширина пика ≈ {width:.3f} Гц; "
            f"доля спектральной энергии около пика ≈ {100 * frac:.1f}%. "
            "Узкий пик → ближе к периодике; широкий → хаос/шум."
        ),
    }


def ляпунов_из_ряда(t: np.ndarray, omega: np.ndarray, eps0: float = 1e-3) -> dict[str, Any]:
    """Грубая оценка max Lyapunov по расхождению близких отрезков Ω(t).

    Не замена метода Бенеттина на полной ОДУ, но даёт знак/порядок для эксперимента.
    """
    ok = np.isfinite(omega) & np.isfinite(t)
    if ok.sum() < 200:
        return {
            "lambda_1_1_с": None,
            "T_пред_с": None,
            "описание": "Ляпунов не посчитан: нужно ≥200 кадров с треком.",
        }
    o = omega[ok]
    tt = t[ok]
    dt = float(np.median(np.diff(tt)))
    # берём пары отрезков, близких по начальному Ω
    win = max(20, int(0.5 / dt))
    divergences = []
    step = max(5, win // 4)
    for i in range(0, len(o) - 2 * win, step):
        for j in range(i + win, len(o) - win, step):
            if abs(o[i] - o[j]) < eps0 * 5 and abs(o[i] - o[j]) > 1e-6:
                d0 = abs(o[i] - o[j]) + 1e-12
                d1 = abs(o[i + win] - o[j + win]) + 1e-12
                divergences.append(np.log(d1 / d0) / (win * dt))
            if len(divergences) > 80:
                break
        if len(divergences) > 80:
            break
    if not divergences:
        return {
            "lambda_1_1_с": None,
            "T_пред_с": None,
            "описание": "Ляпунов: не удалось найти близкие пары отрезков.",
        }
    lam = float(np.median(divergences))
    T = (1.0 / lam) if lam > 1e-4 else None
    return {
        "lambda_1_1_с": lam,
        "T_пред_с": T,
        "описание": (
            f"Оценка показателя Ляпунова λ₁ ≈ {lam:.3f} 1/с "
            f"(>0 → чувствительность к начальным условиям / хаос). "
            + (
                f"Горизонт предсказуемости T≈1/λ₁≈{T:.2f} с."
                if T is not None
                else "λ₁≲0 → движение ближе к регулярному, горизонт формально велик."
            )
        ),
    }


def автокорр_период(t: np.ndarray, omega: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(omega) & np.isfinite(t)
    if ok.sum() < 64:
        return {"T_период_с": None, "описание": "Период по автокорреляции: мало данных."}
    o = omega[ok] - np.nanmean(omega[ok])
    dt = float(np.median(np.diff(t[ok])))
    ac = np.correlate(o, o, mode="full")
    ac = ac[len(ac) // 2 :]
    ac = ac / (ac[0] + 1e-15)
    # первый локальный максимум после нуля
    for i in range(2, min(len(ac) - 1, int(5.0 / dt))):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.2:
            T = i * dt
            return {
                "T_период_с": float(T),
                "описание": (
                    f"Оценка периода по автокорреляции Ω: T≈{T:.3f} с "
                    f"(частота ≈{1 / T:.3f} Гц)."
                ),
            }
    return {
        "T_период_с": None,
        "описание": "Явного периода по автокорреляции не видно (шум/хаос/короткий ряд).",
    }


def статистика_меток(табл: dict[str, np.ndarray]) -> dict[str, Any]:
    n = табл["n_меток"]
    q = табл["качество"]
    total = max(len(n), 1)
    counts = {k: float(np.mean(n == k)) for k in range(0, 4)}
    return {
        "доля_0_меток": counts[0],
        "доля_1_метки": counts[1],
        "доля_2_меток": counts[2],
        "доля_3_меток": counts[3],
        "качество_среднее": float(np.nanmean(q)),
        "качество_медиана": float(np.nanmedian(q)),
        "описание": (
            "Доли кадров по числу видимых ArUco: "
            f"0→{100 * counts[0]:.0f}%, 1→{100 * counts[1]:.0f}%, "
            f"2→{100 * counts[2]:.0f}%, 3→{100 * counts[3]:.0f}%. "
            f"Среднее качество посадки центра Q={float(np.nanmean(q)):.2f} "
            "(1 = метки хорошо согласованы с раскладкой)."
        ),
    }


def оценка_R_с_меток(
    кадры,
    раскладка: dict[str, Any],
) -> dict[str, Any]:
    """Оценка масштаба: сторона ArUco в px → мм/px; контроль r_m между метками."""
    a_мм = float(раскладка.get("сторона_aruco_мм") or 0)
    r_m = float(раскладка.get("радиус_меток_мм") or 0)
    R_yaml = раскладка.get("R_мм")
    sides = []
    dists = []
    углы = {int(k): float(v) for k, v in (раскладка.get("углы_меток_град") or {}).items()}
    for k in кадры:
        if k.n_меток < 1 or not k.мм_на_px:
            continue
        # сторона из углов меток
        for id_, pts in k.углы_меток_px.items():
            d = np.linalg.norm(pts[0] - pts[1])
            sides.append(d * k.мм_на_px)  # должно ≈ a_мм
        ids = list(k.центры_меток_px.keys())
        if len(ids) >= 2 and k.мм_на_px:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if ids[i] not in углы or ids[j] not in углы:
                        continue
                    c1 = np.array(k.центры_меток_px[ids[i]])
                    c2 = np.array(k.центры_меток_px[ids[j]])
                    d_мм = np.linalg.norm(c1 - c2) * k.мм_на_px
                    th1, th2 = np.radians(углы[ids[i]]), np.radians(углы[ids[j]])
                    expect = r_m * np.sqrt(
                        (np.cos(th1) - np.cos(th2)) ** 2 + (np.sin(th1) - np.sin(th2)) ** 2
                    )
                    if expect > 1:
                        dists.append(d_мм / expect)  # ≈1 если ок
    side_med = float(np.median(sides)) if sides else None
    scale_ok = float(np.median(dists)) if dists else None
    R_est = None
    if scale_ok and R_yaml:
        # если расстояния завышены в scale_ok раз относительно раскладки —
        # либо печать не 100%, либо наклейка кривая
        pass
    warn = None
    if side_med is not None and a_мм:
        err = abs(side_med - a_мм) / a_мм
        if err > 0.08:
            warn = (
                f"Сторона метки в кадре ≈{side_med:.2f} мм при ожидаемых {a_мм} мм "
                f"(отклонение {100 * err:.0f}%). Проверьте печать 100%."
            )
    if scale_ok is not None and abs(scale_ok - 1.0) > 0.1:
        warn = (warn or "") + (
            f" Расстояния между метками ×{scale_ok:.2f} относительно схемы — "
            "наклейка/печать могут быть смещены."
        )
    return {
        "сторона_метки_медиана_мм": side_med,
        "сторона_ожидаемая_мм": a_мм,
        "отношение_расстояний_к_схеме": scale_ok,
        "R_yaml_мм": R_yaml,
        "предупреждение": warn,
        "описание": (
            f"Контроль калибровки: медианная сторона ArUco в кадре ≈ "
            f"{side_med:.2f} мм (в yaml {a_мм} мм). "
            + (warn or "Отклонения в пределах нормы.")
            if side_med is not None
            else "Калибровку стороны метки не удалось оценить."
        ),
    }


def полная_сводка(
    табл: dict[str, np.ndarray],
    info: dict[str, Any],
    *,
    кадры=None,
    раскладка: dict | None = None,
) -> dict[str, Any]:
    o = табл["Omega_рад_с"]
    Y = табл["Y_м"]
    Z = табл["Z_м"]
    режим = классифицировать(o, Y)
    sp = спектр_метрики(табл["t_с"], o)
    ly = ляпунов_из_ряда(табл["t_с"], o)
    per = автокорр_период(табл["t_с"], o)
    marks = статистика_меток(табл)
    poinc = сечение_пуанкаре(табл["t_с"], Y, Z, o)
    calib = оценка_R_с_меток(кадры or [], раскладка or {}) if кадры else {}

    ok = np.isfinite(o)
    base = {
        "режим": режим,
        "режим_описание": (
            "Эвристика по хвосту траектории: "
            "малые σ(Y) и σ(Ω) при заметном |Ω| → устойчивое вращение; "
            "малая σ(Ω) → периодика; иначе → хаос/нерегулярность. "
            f"Сейчас: «{режим}»."
        ),
        "fps": info.get("fps"),
        "длительность_с": float(табл["t_с"][-1]) if len(табл["t_с"]) else 0.0,
        "доля_кадров_с_треком": float(np.mean(табл["маска_трека"])),
        "Omega_среднее": float(np.nanmean(o)) if ok.any() else None,
        "Omega_sigma": float(np.nanstd(o)) if ok.any() else None,
        "Omega_макс": float(np.nanmax(np.abs(o))) if ok.any() else None,
        "Y_sigma": float(np.nanstd(Y)),
        "Z_sigma": float(np.nanstd(Z)),
        "r_cm_макс": float(np.nanmax(табл["r_cm_м"])),
        "f_peak_Гц": sp.get("f_peak_Гц"),
        "ширина_пика_Гц": sp.get("ширина_пика_Гц"),
        "доля_энергии_в_пике": sp.get("доля_энергии_в_пике"),
        "T_период_с": per.get("T_период_с"),
        "lambda_1_1_с": ly.get("lambda_1_1_с"),
        "T_пред_с": ly.get("T_пред_с"),
        "n_точек_пуанкаре": int(len(poinc["Omega"])),
        "спектр": sp,
        "ляпунов": ly,
        "период": per,
        "метки": marks,
        "пуанкаре": poinc,
        "калибровка": calib,
    }
    return base
