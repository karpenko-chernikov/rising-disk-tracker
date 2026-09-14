"""Экспорт Excel/CSV и обновление сводки дня."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# русские заголовки для людей
_ЗАГОЛОВКИ = {
    "t_с": "t, с",
    "Y_м": "Y, м",
    "Z_м": "Z, м",
    "Y_сглаж_м": "Y сглаж., м",
    "Z_сглаж_м": "Z сглаж., м",
    "beta_рад": "beta, рад",
    "beta_сглаж_рад": "beta сглаж., рад",
    "Omega_рад_с": "Omega, рад/с",
    "Omega_dot_рад_с2": "dOmega/dt, рад/с²",
    "vY_м_с": "vY, м/с",
    "vZ_м_с": "vZ, м/с",
    "aY_м_с2": "aY, м/с²",
    "aZ_м_с2": "aZ, м/с²",
    "r_cm_м": "r_ЦМ, м",
    "Ek_rot_Дж": "Ek вращ., Дж",
    "Ek_trans_Дж": "Ek поступ., Дж",
    "E_grav_Дж": "E грав. (прокси), Дж",
    "качество": "качество трека",
    "n_меток": "число меток",
    "cx_px": "cx, px",
    "cy_px": "cy, px",
    "маска_трека": "есть трек",
}


def сохранить_ряд(табл: dict[str, Any], out_dir: Path) -> None:
    df = pd.DataFrame(табл)
    # CSV — машинные имена колонок
    df.to_csv(out_dir / "ряд_по_времени.csv", index=False, float_format="%.8g")
    # Excel — русские заголовки
    df_ru = df.rename(columns={k: _ЗАГОЛОВКИ.get(k, k) for k in df.columns})
    with pd.ExcelWriter(out_dir / "ряд_по_времени.xlsx", engine="openpyxl") as w:
        df_ru.to_excel(w, sheet_name="ряд", index=False)
        pd.DataFrame(
            [{"колонка": v, "ключ": k} for k, v in _ЗАГОЛОВКИ.items()]
        ).to_excel(w, sheet_name="словарь", index=False)


def обновить_сводку(день_dir: Path, строка: dict[str, Any]) -> None:
    path = день_dir / "сводка.xlsx"
    new = pd.DataFrame([строка])
    if path.exists():
        old = pd.read_excel(path)
        # заменить если тот же ролик
        if "ролик" in old.columns:
            old = old[old["ролик"] != строка.get("ролик")]
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df.to_excel(path, index=False)
