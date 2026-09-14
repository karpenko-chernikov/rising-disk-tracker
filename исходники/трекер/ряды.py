"""Производные ряды: Ω, энергии, спектр, классификация режима."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from трекер.трек import КадрПоза

G = 9.81


def _savgol_safe(y: np.ndarray, fps: float, window_s: float = 0.25) -> np.ndarray:
    n = len(y)
    if n < 5:
        return y.copy()
    w = int(window_s * fps)
    if w % 2 == 0:
        w += 1
    w = max(5, min(w, n if n % 2 == 1 else n - 1))
    if w < 5:
        return y.copy()
    return savgol_filter(y, w, min(3, w - 1), mode="interp")


def построить_таблицу(
    кадры: list[КадрПоза],
    *,
    fps: float,
    масса_г: float | None,
    R_мм: float | None,
) -> dict[str, np.ndarray]:
    n = len(кадры)
    t = np.array([k.t for k in кадры], dtype=float)
    Y = np.array([np.nan if k.Y_м is None else k.Y_м for k in кадры])
    Z = np.array([np.nan if k.Z_м is None else k.Z_м for k in кадры])
    beta = np.array([np.nan if k.beta_рад is None else k.beta_рад for k in кадры])
    q = np.array([k.quality for k in кадры], dtype=float)
    n_m = np.array([k.n_меток for k in кадры], dtype=float)
    cx = np.array([np.nan if k.cx_px is None else k.cx_px for k in кадры])
    cy = np.array([np.nan if k.cy_px is None else k.cy_px for k in кадры])

    # интерполяция коротких дыр для производных
    def fill(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        ok = np.isfinite(out)
        if ok.sum() < 2:
            return out
        idx = np.arange(n)
        out[~ok] = np.interp(idx[~ok], idx[ok], out[ok])
        return out

    Yf, Zf, bf = fill(Y), fill(Z), fill(beta)
    Ys = _savgol_safe(Yf, fps)
    Zs = _savgol_safe(Zf, fps)
    bs = _savgol_safe(bf, fps)

    dt = 1.0 / fps
    Omega = np.gradient(bs, dt)
    Omega = _savgol_safe(Omega, fps, window_s=0.35)
    Omega_dot = np.gradient(Omega, dt)
    vY = np.gradient(Ys, dt)
    vZ = np.gradient(Zs, dt)
    aY = np.gradient(vY, dt)
    aZ = np.gradient(vZ, dt)
    r_cm = np.hypot(Ys, Zs)

    m = None if масса_г is None else float(масса_г) / 1000.0
    R = None if R_мм is None else float(R_мм) / 1000.0
    if m is not None and R is not None:
        I = 0.5 * m * R * R
        Ek_rot = 0.5 * I * Omega**2
        Ek_tr = 0.5 * m * (vY**2 + vZ**2)
        E_g = m * G * Zs
    else:
        Ek_rot = np.full(n, np.nan)
        Ek_tr = np.full(n, np.nan)
        E_g = np.full(n, np.nan)

    # где не было детекции — NaN в «сырых», но производные на fill
    mask = np.isfinite(Y) & np.isfinite(Z)
    Omega_out = Omega.copy()
    Omega_out[~mask] = np.nan

    return {
        "t_с": t,
        "Y_м": Y,
        "Z_м": Z,
        "Y_сглаж_м": Ys,
        "Z_сглаж_м": Zs,
        "beta_рад": beta,
        "beta_сглаж_рад": bs,
        "Omega_рад_с": Omega_out,
        "Omega_dot_рад_с2": Omega_dot,
        "vY_м_с": vY,
        "vZ_м_с": vZ,
        "aY_м_с2": aY,
        "aZ_м_с2": aZ,
        "r_cm_м": r_cm,
        "Ek_rot_Дж": Ek_rot,
        "Ek_trans_Дж": Ek_tr,
        "E_grav_Дж": E_g,
        "качество": q,
        "n_меток": n_m,
        "cx_px": cx,
        "cy_px": cy,
        "маска_трека": mask.astype(float),
    }


def спектр_omega(t: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(omega) & np.isfinite(t)
    if ok.sum() < 16:
        return np.array([]), np.array([])
    # хвост 80%
    idx = np.where(ok)[0]
    cut = idx[len(idx) // 5 :]
    y = omega[cut] - np.nanmean(omega[cut])
    dt = np.median(np.diff(t[cut]))
    if not np.isfinite(dt) or dt <= 0:
        return np.array([]), np.array([])
    spec = np.fft.rfft(y * np.hanning(len(y)))
    freq = np.fft.rfftfreq(len(y), d=dt)
    amp = np.abs(spec)
    return freq, amp


def классифицировать(omega: np.ndarray, Y: np.ndarray) -> str:
    ok = np.isfinite(omega) & np.isfinite(Y)
    if ok.sum() < 10:
        return "мало данных"
    o = omega[ok]
    y = Y[ok]
    n = len(o)
    tail_o = o[3 * n // 4 :]
    tail_y = y[3 * n // 4 :]
    std_o = float(np.std(tail_o))
    std_y = float(np.std(tail_y))
    mean_o = float(np.abs(np.mean(tail_o)))
    if std_y < 0.015 and std_o < 0.08 and mean_o > 0.20:
        return "устойчивое вращение"
    if std_o < 0.18:
        return "периодические колебания"
    return "хаотическое / нерегулярное"


def сводка_ролика(табл: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, Any]:
    o = табл["Omega_рад_с"]
    Y = табл["Y_м"]
    Z = табл["Z_м"]
    q = табл["качество"]
    ok = np.isfinite(o)
    режим = классифицировать(o, Y)
    freq, amp = спектр_omega(табл["t_с"], o)
    f_peak = float(freq[np.argmax(amp)]) if len(amp) else None
    return {
        "режим": режим,
        "fps": info.get("fps"),
        "длительность_с": float(табл["t_с"][-1]) if len(табл["t_с"]) else 0.0,
        "доля_кадров_с_треком": float(np.mean(табл["маска_трека"])),
        "среднее_качество": float(np.nanmean(q)),
        "Omega_среднее": float(np.nanmean(o)) if ok.any() else None,
        "Omega_sigma": float(np.nanstd(o)) if ok.any() else None,
        "Omega_макс": float(np.nanmax(np.abs(o))) if ok.any() else None,
        "Y_sigma": float(np.nanstd(Y)),
        "Z_sigma": float(np.nanstd(Z)),
        "r_cm_макс": float(np.nanmax(табл["r_cm_м"])),
        "f_peak_Гц": f_peak,
    }
