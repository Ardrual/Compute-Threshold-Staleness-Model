from __future__ import annotations

import argparse


from .model import DriftModel, efficiency_multiplier, effective_compute


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--erisk", type=float, required=True, help="Risk cutoff (FLOPs)")
    parser.add_argument("--tau", type=float, required=True, help="Efficiency doubling time (months)")
    parser.add_argument(
        "--update-interval", type=float, default=None, help="Policy update interval (months)"
    )


def _model_from_args(args: argparse.Namespace) -> DriftModel:
    return DriftModel(
        erisk=args.erisk,
        tau=args.tau,
        update_interval=args.update_interval,
    )


def cmd_efficiency(args: argparse.Namespace) -> int:
    a = efficiency_multiplier(args.t, tau=args.tau)
    print(f"{a:.12g}")
    return 0


def cmd_effective_compute(args: argparse.Namespace) -> int:
    a = efficiency_multiplier(args.t, tau=args.tau)
    e = effective_compute(args.c, a)
    print(f"{e:.12g}")
    return 0


def cmd_threshold(args: argparse.Namespace) -> int:
    model = _model_from_args(args)
    print(f"{model.threshold(args.t):.12g}")
    return 0


def cmd_staleness(args: argparse.Namespace) -> int:
    model = _model_from_args(args)
    print(f"{model.staleness(args.t):.12g}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ctsm", description="Compute Threshold Staleness Model CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eff = sub.add_parser("efficiency", help="Compute A(t)")
    p_eff.add_argument("--t", type=float, required=True, help="Time in months")
    p_eff.add_argument("--tau", type=float, required=True, help="Efficiency doubling time (months)")
    p_eff.set_defaults(func=cmd_efficiency)

    p_ec = sub.add_parser("effective-compute", help="Compute E = A(t)*C")
    p_ec.add_argument("--t", type=float, required=True, help="Time in months")
    p_ec.add_argument("--c", type=float, required=True, help="Raw compute (FLOPs)")
    p_ec.add_argument("--tau", type=float, required=True, help="Efficiency doubling time (months)")
    p_ec.set_defaults(func=cmd_effective_compute)

    p_th = sub.add_parser("threshold", help="Compute policy threshold T(t)")
    p_th.add_argument("--t", type=float, required=True, help="Time in months")
    _add_model_args(p_th)
    p_th.set_defaults(func=cmd_threshold)

    p_st = sub.add_parser("staleness", help="Compute staleness at time t")
    p_st.add_argument("--t", type=float, required=True, help="Time in months")
    _add_model_args(p_st)
    p_st.set_defaults(func=cmd_staleness)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
