"""Sentinel CLI.

    sentinel ask      "some state" --noul "This is urgent"
    sentinel choice   "cancel my subscription" --option safe:"reads" --option destructive:"costs money"
    sentinel score    "critical auth bypass" --levels low medium high critical
    sentinel gate     browse_click '{"selector": "place your order"}'
    sentinel bench    --backend ollama
    sentinel doctor
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Sequence

from . import __version__
from .classify import Classifier
from .embed import HashingEmbedder, describe, default_embedder
from .policy import ClassifierMode, Policy, Risk
from .types import Choice, Noul, Score


def _build(args: argparse.Namespace) -> Classifier:
    if getattr(args, "backend", "auto") == "hashing":
        return Classifier(HashingEmbedder())
    return Classifier(default_embedder(prefer_ollama=getattr(args, "backend", "auto") != "hashing"))


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_ask(args: argparse.Namespace) -> int:
    clf = _build(args)
    ans = clf.ask(args.state, Noul(args.noul))
    _emit(
        {
            "state": args.state,
            "question": args.noul,
            "value": ans.value,
            "probabilities": ans.probabilities,
            "confidence": ans.confidence,
            "calibrated": ans.calibrated,
            "embedder": describe(clf.embedder),
        }
    )
    return 0


def cmd_choice(args: argparse.Namespace) -> int:
    clf = _build(args)
    options = []
    for spec in args.option:
        label, _, description = spec.partition(":")
        options.append((label.strip(), description.strip() or label.strip()))
    ans = clf.ask(args.state, Choice(options=options, instructions=args.instructions))
    _emit(
        {
            "state": args.state,
            "value": ans.value,
            "probabilities": ans.probabilities,
            "confidence": ans.confidence,
            "calibrated": ans.calibrated,
        }
    )
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    clf = _build(args)
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    ans = clf.ask(args.state, Score(labels=levels))
    _emit(
        {
            "state": args.state,
            "value": ans.value,
            "top": max(ans.probabilities.items(), key=lambda kv: kv[1])[0],
            "probabilities": ans.probabilities,
            "confidence": ans.confidence,
            "calibrated": ans.calibrated,
        }
    )
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    try:
        call_args = json.loads(args.args_json) if args.args_json else {}
    except json.JSONDecodeError as exc:
        print(f"error: --args is not valid JSON: {exc}", file=sys.stderr)
        return 2
    policy = Policy(_build(args), min_confidence=args.min_confidence)
    decision = policy.evaluate(args.tool, call_args)
    _emit(
        {
            "tool": args.tool,
            "args": call_args,
            "needs_human": decision.needs_human,
            "risk": decision.risk.value,
            "confidence": decision.confidence,
            "floor": decision.floor.value,
            "reason": decision.reason,
            "degraded": decision.degraded,
        }
    )
    # exit 0 = may proceed, 10 = needs a human. Scriptable.
    return 10 if decision.needs_human else 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Steady-state per-decision latency, the number the gate is about.

    The first `evaluate` call pays for loading the model and embedding the four
    risk prototypes; every later call embeds only the action text, because the
    prototypes are cached. So this reports cold and warm separately rather than
    averaging them into one number that means nothing.
    """
    from .adversarial import ADVERSARIAL_SET
    from .eval_report import EVAL_SET

    clf = _build(args)
    policy = Policy(clf, classifier_mode=ClassifierMode(args.mode))
    cases = [*EVAL_SET, *ADVERSARIAL_SET]

    t0 = time.perf_counter()
    policy.evaluate(cases[0][0], cases[0][1])
    cold_ms = (time.perf_counter() - t0) * 1000

    # warm-up: fill the prototype cache and the daemon's model slot
    for tool, args_, _ in cases:
        policy.evaluate(tool, args_)

    warm: list[float] = []
    for round_no in range(args.rounds):
        for tool, args_, _ in cases:
            # A nonce per round, so every decision is a genuine cache miss.
            # Without it the classifier's own text cache serves the second round
            # onwards and the benchmark reports 1 ms for a decision that really
            # costs an embedding. The risk prototypes stay cached, which is what
            # happens in production.
            call = dict(args_)
            if "selector" in call:
                call["selector"] = f"{call['selector']} n{round_no}"
            t = time.perf_counter()
            policy.evaluate(tool, call)
            warm.append((time.perf_counter() - t) * 1000)

    def pct(values: Sequence[float], p: float) -> float:
        ordered = sorted(values)
        return ordered[min(int(len(ordered) * p), len(ordered) - 1)]

    _emit(
        {
            "embedder": describe(clf.embedder),
            "batch_endpoint": getattr(clf.embedder, "_batch_ok", None),
            "decisions": len(cases) * args.rounds,
            "cold_first_call_ms": round(cold_ms, 2),
            "warm_ms": {
                "p50": round(pct(warm, 0.50), 2),
                "p95": round(pct(warm, 0.95), 2),
                "p99": round(pct(warm, 0.99), 2),
                "max": round(max(warm), 2),
                "mean": round(sum(warm) / len(warm), 2),
            },
            "gate": {
                "target_p95_ms": args.target,
                "p95_ms": round(pct(warm, 0.95), 2),
                "pass": pct(warm, 0.95) <= args.target,
            },
        }
    )
    return 0 if pct(warm, 0.95) <= args.target else 1


def cmd_eval(args: argparse.Namespace) -> int:
    from .embed import describe
    from .eval_report import full_report

    emb = default_embedder(prefer_ollama=args.backend != "hashing")
    report = full_report(
        emb,
        min_confidence=args.min_confidence,
        classifier_mode=ClassifierMode(args.mode),
    )
    _emit(report)
    dev, adv = report["dev_set"], report["adversarial_set"]
    # Non-zero if anything unsafe was allowed on either set. Friction alone is
    # reported but is not a build failure.
    return 1 if (dev["allowed_unsafely"] or adv["allowed_unsafely"]) else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    emb = default_embedder(prefer_ollama=not args.backend == "hashing")
    degraded = getattr(emb, "degraded", False)
    _emit(
        {
            "version": __version__,
            "embedder": describe(emb),
            "degraded": bool(degraded),
            "reason": getattr(emb, "reason", None),
            "dim": emb.dim,
        }
    )
    return 1 if degraded else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sentinel", description="Local action gating.")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--backend", choices=("auto", "ollama", "hashing"), default="auto")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("ask", help="yes/no question about a state")
    a.add_argument("state")
    a.add_argument("--noul", required=True, help="the property being judged")
    a.set_defaults(func=cmd_ask)

    c = sub.add_parser("choice", help="pick one of several options")
    c.add_argument("state")
    c.add_argument("--option", action="append", required=True, metavar="LABEL:DESC")
    c.add_argument("--instructions", default="")
    c.set_defaults(func=cmd_choice)

    s = sub.add_parser("score", help="rate against ordered levels")
    s.add_argument("state")
    s.add_argument("--levels", required=True, help="comma separated, low to high")
    s.set_defaults(func=cmd_score)

    g = sub.add_parser("gate", help="decide whether a tool call may proceed")
    g.add_argument("tool")
    g.add_argument("--args", dest="args_json", default="")
    g.add_argument("--min-confidence", type=float, default=0.12)
    g.set_defaults(func=cmd_gate)

    b = sub.add_parser("bench", help="steady-state per-decision latency")
    b.add_argument("--rounds", type=int, default=5)
    b.add_argument("--target", type=float, default=50.0, help="p95 gate in ms")
    b.add_argument("--mode", choices=("off", "advisory", "blocking"), default="advisory")
    b.set_defaults(func=cmd_bench)

    e = sub.add_parser("eval", help="confusion matrix + separability on the labelled set")
    e.add_argument("--min-confidence", type=float, default=0.12)
    e.add_argument("--mode", choices=("off", "advisory", "blocking"), default="advisory")
    e.set_defaults(func=cmd_eval)

    d = sub.add_parser("doctor", help="check the embedding backend")
    d.set_defaults(func=cmd_doctor)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
