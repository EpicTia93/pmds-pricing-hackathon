"""Command line: ``uv run python -m swarm {evolve,tournament,export,params}``."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

from .evolve import Evolution, load_experiment, load_param_sets, opponent_entrants, scenario_pool
from .export import export_bundle
from .presets import PRESETS, resolve as resolve_presets
from .space import SPECS, ParamSpace
from .strategy import DEFAULT_PARAMS
from .tournament import bundle_entrant, leaderboard, run_matches, schedule, score


def _set_overrides(pairs: list[str]) -> dict:
    """Parse ``--set a.b=value`` pairs (values are YAML) into a nested dict."""
    out: dict = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return out


def cmd_evolve(args) -> None:
    exp = load_experiment(args.experiment, _set_overrides(args.set))
    champion = Evolution(exp).run()
    if args.export:
        params = load_param_sets(champion)[0]
        print(f"Exported bundle: {export_bundle(params, args.export, str(champion), overwrite=True)}")


def cmd_tournament(args) -> None:
    exp = load_experiment(args.experiment, _set_overrides(args.set))
    entrants = []
    for name, params in resolve_presets(args.preset or None).items():
        entrants.append({"id": f"preset:{name}", "kind": "param", "params": params})
    for path in args.params or []:
        sets = load_param_sets(path)[: args.top]
        stem = Path(path).parent.name + "/" + Path(path).stem
        for i, params in enumerate(sets):
            entrant_id = stem if len(sets) == 1 else f"{stem}#{i + 1}"
            entrants.append({"id": entrant_id, "kind": "param", "params": {**DEFAULT_PARAMS, **params}})
    rng = random.Random(args.seed)
    for _ in range(args.random):
        entrants.append({"id": f"random{len(entrants)}", "kind": "param",
                         "params": ParamSpace(exp["space"], exp["fixed"]).sample(rng)})
    entrants += [bundle_entrant(path) for path in args.bundle or []]
    if not entrants:
        sys.exit("no entrants: use --preset, --params, --bundle or --random")
    size = args.match_size or len(entrants)
    jobs = schedule(entrants, size, args.rounds, scenario_pool(exp), rng,
                    fixed_opponents=opponent_entrants(exp))
    print(f"{len(entrants)} entrants, {len(jobs)} matches")
    records = score(run_matches(jobs, exp["workers"]))
    print(leaderboard(records))
    if args.save:
        ids = {e["id"]: e for e in entrants}
        ranked = sorted((t for t in records if t in ids and ids[t]["kind"] == "param"),
                        key=lambda t: records[t].fitness)
        Path(args.save).write_text(json.dumps(
            [{"id": t, "params": ids[t]["params"], **records[t].summary()} for t in ranked], indent=1))
        print(f"saved ranking to {args.save}")


def cmd_export(args) -> None:
    if args.preset:
        params, source = PRESETS[args.preset], f"preset {args.preset}"
    else:
        params, source = load_param_sets(args.params)[args.index], args.params
    print(export_bundle(params, args.name, source, overwrite=args.overwrite))


def cmd_params(args) -> None:
    print(f"{'parameter':<20} {'default':>12}  space")
    for name, default in DEFAULT_PARAMS.items():
        spec = SPECS[name]
        rng = (f"{spec.choices}" if spec.kind == "choice" else
               "bool" if spec.kind == "bool" else
               f"{spec.kind} [{spec.low}, {spec.high}]{' log' if spec.log else ''}")
        print(f"{name:<20} {str(default):>12}  {rng}")
    print(f"\npresets: {', '.join(PRESETS)}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m swarm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--experiment", "-e", help="experiment YAML (scenarios, opponents, ...)")
    common.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override experiment keys, e.g. --set population=64 --set selection.elite=8")

    p = sub.add_parser("evolve", parents=[common], help="run an evolutionary search")
    p.add_argument("--export", metavar="NAME", help="export the champion as agent bundle NAME")
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("tournament", parents=[common], help="pit fixed strategies against each other")
    p.add_argument("--preset", nargs="*", help="preset names, or 'all'")
    p.add_argument("--params", nargs="*", help="champion/generation/playoff JSON files")
    p.add_argument("--top", type=int, default=5, help="entries taken from each multi-entry file")
    p.add_argument("--bundle", nargs="*", help="agent bundle folders")
    p.add_argument("--random", type=int, default=0, help="add N random strategies")
    p.add_argument("--rounds", type=int, default=20, help="matches per entrant")
    p.add_argument("--match-size", type=int, help="entrants per match (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", help="write ranked params JSON")
    p.set_defaults(func=cmd_tournament)

    p = sub.add_parser("export", help="write an uploadable agent bundle")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--params", help="champion/generation/playoff JSON")
    src.add_argument("--preset")
    p.add_argument("--index", type=int, default=0, help="entry index in a multi-entry file")
    p.add_argument("--name", required=True, help="bundle folder name under src/dme_simulator/agents")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("params", help="list parameters, defaults and search ranges")
    p.set_defaults(func=cmd_params)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
