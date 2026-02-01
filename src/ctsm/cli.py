from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .model import DriftModel, efficiency_multiplier, effective_compute, metrics_from_arrays


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--erisk", type=float, required=True, help="Risk cutoff (FLOPs)")
    parser.add_argument("--tau", type=float, required=True, help="Efficiency doubling time (months)")
    parser.add_argument(
        "--update-interval", type=float, default=None, help="Policy update interval (months)"
    )


def _add_dist_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mu0", type=float, default=57.0, help="Baseline mean log compute")
    parser.add_argument("--r", type=float, default=0.12, help="Growth rate of mean log compute")
    parser.add_argument("--sigma", type=float, default=2.0, help="Std dev of log compute")


def _add_scenario_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scenario",
        type=str,
        default="baseline",
        help="Scenario: baseline or threshold-sensitive",
    )
    parser.add_argument(
        "--strategic-share",
        type=float,
        default=0.0,
        help="Share of threshold-sensitive runs (p in the mixture)",
    )
    parser.add_argument(
        "--strategic-beta",
        type=float,
        default=0.9,
        help="Safety margin beta for strategic runs (fraction of threshold)",
    )
    parser.add_argument(
        "--strategic-sigma",
        type=float,
        default=1.0,
        help="Std dev of log compute for strategic runs",
    )


def _model_from_args(args: argparse.Namespace) -> DriftModel:
    return DriftModel(
        erisk=args.erisk,
        tau=args.tau,
        update_interval=args.update_interval,
        mu0=getattr(args, "mu0", 57.0),
        r=getattr(args, "r", 0.12),
        sigma=getattr(args, "sigma", 2.0),
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


def cmd_simulate(args: argparse.Namespace) -> int:
    model = _model_from_args(args)
    rng = np.random.default_rng(args.seed)
    metrics = model.simulate_metrics(
        args.t,
        args.n,
        rng=rng,
        scenario=args.scenario,
        strategic_share=args.strategic_share,
        strategic_beta=args.strategic_beta,
        strategic_sigma=args.strategic_sigma,
    )
    if args.json:
        print(json.dumps(metrics, sort_keys=True))
    else:
        print(f"fnr={metrics['fnr']:.6g} coverage={metrics['coverage']:.6g}")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    c = np.asarray(args.c, dtype=float)
    metrics = metrics_from_arrays(c, args.a, args.erisk, args.threshold)
    if args.json:
        print(json.dumps(metrics, sort_keys=True))
    else:
        print(f"fnr={metrics['fnr']:.6g} coverage={metrics['coverage']:.6g}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="efdm", description="Compute Threshold Staleness Model CLI")
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

    p_sim = sub.add_parser("simulate", help="Monte Carlo estimate FNR/FPR/Coverage")
    p_sim.add_argument("--t", type=float, required=True, help="Time in months")
    p_sim.add_argument("--n", type=int, required=True, help="Number of samples")
    p_sim.add_argument("--seed", type=int, default=None, help="RNG seed")
    p_sim.add_argument("--json", action="store_true", help="Output JSON")
    _add_model_args(p_sim)
    _add_dist_args(p_sim)
    _add_scenario_args(p_sim)
    p_sim.set_defaults(func=cmd_simulate)

    p_m = sub.add_parser("metrics", help="Compute metrics from explicit samples")
    p_m.add_argument("--a", type=float, required=True, help="Efficiency multiplier A(t)")
    p_m.add_argument("--erisk", type=float, required=True, help="Risk cutoff (FLOPs)")
    p_m.add_argument("--threshold", type=float, required=True, help="Policy threshold T(t)")
    p_m.add_argument("--c", type=float, nargs="+", required=True, help="Compute samples (FLOPs)")
    p_m.add_argument("--json", action="store_true", help="Output JSON")
    p_m.set_defaults(func=cmd_metrics)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
