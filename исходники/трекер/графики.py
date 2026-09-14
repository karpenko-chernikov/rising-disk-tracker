"""Статические графики (тёмная тема) с понятными подписями."""

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


def сохранить_графики(
    табл: dict[str, np.ndarray],
    out: Path,
    *,
    заголовок: str = "",
    сводка: dict[str, Any] | None = None,
    модель: dict[str, Any] | None = None,
) -> None:
    t = табл["t_с"]
    Y, Z = табл["Y_м"], табл["Z_м"]
    Om = табл["Omega_рад_с"]
    q = табл["качество"]
    сводка = сводка or {}

    # траектория YZ
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    x, y = _finite(Y, Z)
    if len(x):
        _glow(ax, x, y, CYAN)
        ax.scatter([x[0]], [y[0]], c=YELL, s=40, zorder=5, label="старт")
        ax.scatter([x[-1]], [y[-1]], c=CYAN, s=50, marker="D", zorder=5, label="финиш")
    ax.set_xlabel("Y, м  (вправо по кадру)")
    ax.set_ylabel("Z, м  (вверх по кадру)")
    ax.set_title("Траектория центра диска (ЦМ)\nЖёлтый=старт, ромб=конец")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(facecolor=BG, edgecolor="#333355", labelcolor=FG)
    if заголовок:
        fig.text(0.5, 0.02, заголовок, ha="center", color=FG, fontsize=8)
    _save(fig, out / "траектория_YZ.png")

    def series(yarr, fname, ylab, color, title, note):
        fig, ax = plt.subplots(figsize=(8, 3.4))
        xx, yy = _finite(t, yarr)
        if len(xx):
            _glow(ax, xx, yy, color)
        ax.set_xlabel("t, с — время от начала ролика")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        fig.text(0.01, 0.01, note, ha="left", va="bottom", color="#8888aa", fontsize=7)
        _save(fig, out / f"{fname}.png")

    series(
        Y, "Y_от_t", "Y, м", CYAN,
        "Y(t) — смещение центра вбок",
        "Нуль Y,Z ≈ первые 2 секунды. Рост |Y| — диск уходит в сторону.",
    )
    series(
        Z, "Z_от_t", "Z, м", GREE,
        "Z(t) — смещение центра вверх/вниз («подъём»)",
        "В модели Maas Z связан с «поднятием»; знак зависит от ориентации камеры.",
    )
    series(
        Om, "Omega_от_t", "Ω, рад/с", ORNG,
        "Ω(t) — угловая скорость ДИСКА",
        "Не путать с n (об/мин ведущего колеса) из имени файла.",
    )
    series(
        табл["beta_сглаж_рад"], "beta_от_t", "β, рад", MGNT,
        "β(t) — угол поворота диска (развёрнутый)",
        "Розовая стрелка на видео указывает направление β.",
    )
    series(
        q, "качество_трека", "Q (0…1)", YELL,
        "Качество посадки центра по меткам",
        "Близко к 1 — метки согласованы; низко — смаз/окклюзия/кривая наклейка.",
    )
    series(
        табл["n_меток"], "число_меток_от_t", "число ArUco", CYAN,
        "Сколько меток видно в кадре",
        "Для надёжного центра нужно ≥2; идеально 3.",
    )

    # фазы
    fig, ax = plt.subplots(figsize=(6, 5))
    x, y = _finite(Y, Om)
    if len(x):
        _glow(ax, x, y, MGNT, lw=1.2)
        ax.scatter([x[0]], [y[0]], c=YELL, s=40, zorder=5)
        ax.scatter([x[-1]], [y[-1]], c=CYAN, s=50, marker="D", zorder=5)
    ax.set_xlabel("Y, м")
    ax.set_ylabel("Ω, рад/с")
    ax.set_title("Фазовый портрет Y–Ω\nЗамкнутая кривая ≈ периодика; клубок ≈ хаос")
    _save(fig, out / "фаза_Y_Omega.png")

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
    fig, ax = plt.subplots(figsize=(8, 3.4))
    if len(freq):
        _glow(ax, freq, amp, CYAN)
        fp = сводка.get("f_peak_Гц")
        if fp:
            ax.axvline(fp, color=ORNG, ls="--", lw=1, label=f"пик {fp:.3f} Гц")
            ax.legend(facecolor=BG, edgecolor="#333355", labelcolor=FG)
    ax.set_xlabel("f, Гц — частота")
    ax.set_ylabel("|FFT Ω| — амплитуда спектра")
    ax.set_title("Спектр угловой скорости Ω(t)\nУзкий пик → ритм; широкий → хаос/шум")
    _save(fig, out / "спектр_Omega.png")

    # энергия
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for key, color, lab in [
        ("Ek_rot_Дж", ORNG, "Ek вращ. ≈ ½IΩ²"),
        ("Ek_trans_Дж", CYAN, "Ek поступ. ≈ ½m v²"),
        ("E_grav_Дж", GREE, "mgZ (прокси подъёма)"),
    ]:
        xx, yy = _finite(t, табл[key])
        if len(xx):
            _glow(ax, xx, yy, color, lw=1.3)
            ax.plot([], [], color=color, label=lab)
    ax.legend(facecolor=BG, edgecolor="#333355", labelcolor=FG, fontsize=8)
    ax.set_xlabel("t, с")
    ax.set_ylabel("E, Дж")
    ax.set_title("Оценки энергий (порядок величины, нуль Z относительный)")
    _save(fig, out / "энергия.png")

    # Пуанкарé
    po = (сводка.get("пуанкаре") or {})
    Om_p = po.get("Omega", np.array([]))
    Z_p = po.get("Z", np.array([]))
    Om_n = po.get("Omega_next", np.array([]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    if len(Om_p):
        ax.scatter(Om_p, Z_p, s=12, c=CYAN, alpha=0.8)
    ax.set_xlabel("Ω при пересечении Y=0, рад/с")
    ax.set_ylabel("Z, м")
    ax.set_title("Сечение Пуанкарé (Y=0 ↑)\nТочки упорядочены → периодика")
    ax = axes[1]
    if len(Om_n):
        ax.scatter(Om_p[: len(Om_n)], Om_n, s=12, c=ORNG, alpha=0.8)
        lim = [np.nanmin(Om_p), np.nanmax(Om_p)]
        ax.plot(lim, lim, color="#555577", lw=1)
    ax.set_xlabel("Ω_n")
    ax.set_ylabel("Ω_{n+1}")
    ax.set_title("Карта возврата Ω\nНа диагонали → период-1")
    _save(fig, out / "пуанкаре.png")

    # heatmap пребывания
    fig, ax = plt.subplots(figsize=(6, 5))
    xx, yy = _finite(Y, Z)
    if len(xx) > 10:
        hb = ax.hexbin(xx, yy, gridsize=25, cmap="magma", mincnt=1)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("число кадров в ячейке")
    ax.set_xlabel("Y, м")
    ax.set_ylabel("Z, м")
    ax.set_title("Где центр бывал чаще (теплокарта)")
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig, out / "теплокарта_YZ.png")

    # сравнение с моделью
    if модель and модель.get("Y_модель") is not None:
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        for ax, key, mkey, lab, color in [
            (axes[0], "Y_м", "Y_модель", "Y, м", CYAN),
            (axes[1], "Z_м", "Z_модель", "Z, м", GREE),
            (axes[2], "Omega_рад_с", "Omega_модель", "Ω, рад/с", ORNG),
        ]:
            x1, y1 = _finite(t, табл[key])
            x2, y2 = _finite(t, модель[mkey])
            if len(x1):
                _glow(ax, x1, y1, color, lw=1.2)
            if len(x2):
                ax.plot(x2, y2, color=YELL, lw=1.2, alpha=0.9, label="модель Maas")
            ax.set_ylabel(lab)
            ax.legend(facecolor=BG, edgecolor="#333355", labelcolor=FG, fontsize=8)
        axes[0].set_title("Эксперимент (цвет) vs модель Maas (жёлтый)")
        axes[2].set_xlabel("t, с")
        fig.text(
            0.01, 0.01,
            "Модель с теми же W, δ, R, m; трение k0/k1/k2 по умолчанию. Невязка ≠ ошибка CV.",
            color="#8888aa", fontsize=7,
        )
        _save(fig, out / "сравнение_с_моделью.png")
