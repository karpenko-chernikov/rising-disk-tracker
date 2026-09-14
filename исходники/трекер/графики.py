"""Статические графики (тёмная неоновая тема)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from трекер.ряды import спектр_omega

BG = "#0a0a14"
FG = "#c8c8d8"
CYAN = "#00e5ff"
YELL = "#ffe600"
ORNG = "#ff6e40"
GREE = "#39ff14"
MGNT = "#ff2bd6"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": "#333355",
        "axes.labelcolor": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "text.color": FG,
        "grid.color": "#1a1a2e",
        "axes.grid": True,
        "font.size": 10,
    }
)


def _glow(ax, x, y, color, lw=1.6):
    ax.plot(x, y, color=color, lw=5, alpha=0.12)
    ax.plot(x, y, color=color, lw=lw)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _finite(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def сохранить_графики(табл: dict[str, np.ndarray], out: Path, заголовок: str = "") -> None:
    t = табл["t_с"]
    Y, Z = табл["Y_м"], табл["Z_м"]
    Om = табл["Omega_рад_с"]
    q = табл["качество"]

    # траектория YZ
    fig, ax = plt.subplots(figsize=(6, 6))
    x, y = _finite(Y, Z)
    if len(x):
        _glow(ax, x, y, CYAN)
        ax.scatter([x[0]], [y[0]], c=YELL, s=40, zorder=5, label="старт")
        ax.scatter([x[-1]], [y[-1]], c=CYAN, s=50, marker="D", zorder=5, label="финиш")
    ax.set_xlabel("Y, м")
    ax.set_ylabel("Z, м")
    ax.set_title(f"Траектория ЦМ\n{заголовок}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(facecolor=BG, edgecolor="#333355", labelcolor=FG)
    _save(fig, out / "траектория_YZ.png")

    def series(y, name, ylab, color):
        fig, ax = plt.subplots(figsize=(8, 3.2))
        xx, yy = _finite(t, y)
        if len(xx):
            _glow(ax, xx, yy, color)
        ax.set_xlabel("t, с")
        ax.set_ylabel(ylab)
        ax.set_title(name)
        _save(fig, out / f"{name}.png")

    series(Y, "Y_от_t", "Y, м", CYAN)
    series(Z, "Z_от_t", "Z, м", GREE)
    series(Om, "Omega_от_t", "Ω, рад/с", ORNG)
    series(табл["beta_сглаж_рад"], "beta_от_t", "β, рад", MGNT)
    series(q, "качество_трека", "качество", YELL)

    # фаза Y–Omega
    fig, ax = plt.subplots(figsize=(6, 5))
    x, y = _finite(Y, Om)
    if len(x):
        _glow(ax, x, y, MGNT, lw=1.2)
        ax.scatter([x[0]], [y[0]], c=YELL, s=40, zorder=5)
        ax.scatter([x[-1]], [y[-1]], c=CYAN, s=50, marker="D", zorder=5)
    ax.set_xlabel("Y, м")
    ax.set_ylabel("Ω, рад/с")
    ax.set_title("Фазовый портрет Y–Ω")
    _save(fig, out / "фаза_Y_Omega.png")

    # фаза YZ уже есть; фаза Z–Omega
    fig, ax = plt.subplots(figsize=(6, 5))
    x, y = _finite(Z, Om)
    if len(x):
        _glow(ax, x, y, GREE, lw=1.2)
    ax.set_xlabel("Z, м")
    ax.set_ylabel("Ω, рад/с")
    ax.set_title("Фазовый портрет Z–Ω")
    _save(fig, out / "фаза_Z_Omega.png")

    # спектр
    freq, amp = спектр_omega(t, Om)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if len(freq):
        _glow(ax, freq, amp, CYAN)
    ax.set_xlabel("f, Гц")
    ax.set_ylabel("|FFT Ω|")
    ax.set_title("Спектр Ω(t)")
    _save(fig, out / "спектр_Omega.png")

    # энергия
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for key, color, lab in [
        ("Ek_rot_Дж", ORNG, "Ek вращ."),
        ("Ek_trans_Дж", CYAN, "Ek поступ."),
        ("E_grav_Дж", GREE, "mgZ"),
    ]:
        xx, yy = _finite(t, табл[key])
        if len(xx):
            _glow(ax, xx, yy, color, lw=1.3)
            ax.plot([], [], color=color, label=lab)
    ax.legend(facecolor=BG, edgecolor="#333355", labelcolor=FG)
    ax.set_xlabel("t, с")
    ax.set_ylabel("E, Дж")
    ax.set_title("Энергии (прокси)")
    _save(fig, out / "энергия.png")

    # превью: один кадр-статистика quality vs t already; still overlay saved separately
