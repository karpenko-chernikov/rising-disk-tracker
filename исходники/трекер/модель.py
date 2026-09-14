"""Упрощённая модель Maas для сравнения с экспериментом."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

G = 9.81
K0, K1, K2 = 0.012, 0.103, 0.055


def _friction(omega: float) -> float:
    return max(K1 - K2 * abs(omega), K0)


def _rhs(t, state, W, delta, r, m):
    """delta — горизонтальный сдвиг контакта (ось Y), м: Y_eff = Y − δ."""
    y, z, omega, beta = state
    y_eff = y - delta
    I = m * (0.5 * r**2 + y_eff**2 + z**2)
    dydt = z * omega
    dzdt = -y_eff * omega + W
    fr = _friction(omega) * omega * m
    drive = m * W * z * omega
    # гравитационный момент ~ горизонтальному рычагу от вертикали через контакт
    domega = (m * G * y_eff - fr - drive) / I
    return [dydt, dzdt, domega, omega]


def смоделировать(
    *,
    W_м_с: float,
    delta_м: float,
    R_м: float,
    масса_кг: float,
    t_end: float,
    Y0: float = 0.0,
    Z0: float = 0.0,
    Omega0: float = 1.0,
) -> dict[str, np.ndarray]:
    if t_end < 0.5 or W_м_с is None:
        return {}
    sol = solve_ivp(
        _rhs,
        (0, t_end),
        [Y0, Z0, Omega0, 0.0],
        args=(W_м_с, delta_м, R_м, масса_кг),
        dense_output=True,
        rtol=1e-7,
        atol=1e-9,
        max_step=0.02,
    )
    if not sol.success:
        return {}
    t = np.linspace(0, t_end, max(50, int(t_end * 60)))
    Y, Z, Om, beta = sol.sol(t)
    return {"t_с": t, "Y_м": Y, "Z_м": Z, "Omega_рад_с": Om, "beta_рад": beta}


def невязка_с_экспериментом(
    табл: dict[str, np.ndarray],
    модель: dict[str, np.ndarray],
) -> dict[str, Any]:
    if not модель:
        return {
            "rmse_Y": None,
            "rmse_Z": None,
            "rmse_Omega": None,
            "описание": "Модель не посчитана (нет W/δ/R/m или короткий ролик).",
        }
    t = табл["t_с"]
    def interp(key):
        return np.interp(t, модель["t_с"], модель[key], left=np.nan, right=np.nan)

    Ym, Zm, Om = interp("Y_м"), interp("Z_м"), interp("Omega_рад_с")
    mask = np.isfinite(табл["Y_м"]) & np.isfinite(Ym)
    if mask.sum() < 10:
        return {"rmse_Y": None, "rmse_Z": None, "rmse_Omega": None, "описание": "Мало общего времени для сравнения."}

    def rmse(a, b, m):
        return float(np.sqrt(np.nanmean((a[m] - b[m]) ** 2)))

    m2 = mask & np.isfinite(табл["Omega_рад_с"])
    rY = rmse(табл["Y_м"], Ym, mask)
    rZ = rmse(табл["Z_м"], Zm, mask)
    rO = rmse(табл["Omega_рад_с"], Om, m2) if m2.sum() else None
    return {
        "rmse_Y": rY,
        "rmse_Z": rZ,
        "rmse_Omega": rO,
        "Y_модель": Ym,
        "Z_модель": Zm,
        "Omega_модель": Om,
        "описание": (
            "Сравнение с размерной моделью Maas (те же W, δ, R, m; трение k0/k1/k2 по умолчанию). "
            f"RMSE: Y={rY*1000:.2f} мм, Z={rZ*1000:.2f} мм"
            + (f", Ω={rO:.3f} рад/с." if rO is not None else ".")
            + " Большая невязка ≠ ошибка трекера: модель упрощена; δ у нас — горизонтальный сдвиг контакта."
        ),
    }
