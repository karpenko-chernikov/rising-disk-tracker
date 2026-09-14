"""Трекер поднимающегося диска — CLI для команды Бобры."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ИСХОДНИКИ = Path(__file__).resolve().parents[1]
КОРЕНЬ = _ИСХОДНИКИ.parent  # tracker/
if str(_ИСХОДНИКИ) not in sys.path:
    sys.path.insert(0, str(_ИСХОДНИКИ))


def _cmd_метки(args: argparse.Namespace) -> int:
    from трекер.метки import запустить_метки

    return запустить_метки(
        r_мм=args.R_мм,
        диаметр_мм=args.диаметр_мм,
        масса_г=args.масса_г,
        id_диска=args.id,
        интерактив=args.интерактив,
        корень=КОРЕНЬ,
    )


def _cmd_обработать(args: argparse.Namespace) -> int:
    from трекер.обработка import запустить_обработку

    return запустить_обработку(корень=КОРЕНЬ, только=args.файл)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="трекер",
        description="Трекер «поднимающийся диск»: метки ArUco и обработка видео",
    )
    sub = parser.add_subparsers(dest="команда", required=True)

    p_метки = sub.add_parser("метки", help="Спросить R/массу, напечатать PDF меток + yaml")
    p_метки.add_argument("--R-мм", dest="R_мм", type=float, default=None)
    p_метки.add_argument("--диаметр-мм", dest="диаметр_мм", type=float, default=None)
    p_метки.add_argument("--масса-г", dest="масса_г", type=float, default=None)
    p_метки.add_argument("--id", dest="id", type=str, default=None)
    p_метки.add_argument(
        "--интерактив",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Спрашивать недостающие поля в терминале (по умолчанию да)",
    )
    p_метки.set_defaults(func=_cmd_метки)

    p_обр = sub.add_parser("обработать", help="Разобрать видео из папки входящие/")
    p_обр.add_argument("--файл", type=str, default=None, help="Один файл вместо всей очереди")
    p_обр.set_defaults(func=_cmd_обработать)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
