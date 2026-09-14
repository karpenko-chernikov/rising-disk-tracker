"""Разбор имени файла: n120_d-10.mov → об/мин и δ мм."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class ПараметрыИмени:
    n_об_мин: float | None = None
    delta_мм: float | None = None
    сырое_имя: str = ""


_N = re.compile(r"(?i)(?:^|[_\s-])n\s*([+-]?\d+(?:[.,]\d+)?)")
_D = re.compile(r"(?i)(?:^|[_\s-])(?:d|δ|delta)\s*([+-]?\d+(?:[.,]\d+)?)")


def разобрать_имя(имя_файла: str) -> ПараметрыИмени:
    stem = имя_файла.rsplit(".", 1)[0] if "." in имя_файла else имя_файла
    s = stem.replace(" ", "_")
    out = ПараметрыИмени(сырое_имя=stem)

    m = _N.search("_" + s)
    if m:
        out.n_об_мин = float(m.group(1).replace(",", "."))

    m = _D.search("_" + s)
    if m:
        out.delta_мм = float(m.group(1).replace(",", "."))

    return out


def w_из_n(n_об_мин: float, r_кол_мм: float) -> float:
    """Линейный W в м/с: W = ω · r_кол."""
    omega = 2.0 * math.pi * n_об_мин / 60.0
    return omega * (r_кол_мм / 1000.0)
