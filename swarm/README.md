# Strategy swarm

A parametrised agent (`strategy.py`), a parallel tournament runner and an
evolutionary search that picks the strongest parameter sets. It lives outside
`src/dme_simulator/agents/`, so none of it is auto-discovered by `dme-sim` until
you **export** a champion.

| File | Role |
| --- | --- |
| `strategy.py` | `ParamAgent` plus `DEFAULT_PARAMS`, which holds every knob. Self-contained, so it is copied into exported bundles. |
| `space.py` | Search range for each parameter; sample, mutate, crossover. |
| `presets.py` | Hand-made archetypes that seed the population. |
| `scenarios.py` | Loads scenarios and builds randomly perturbed variants of them (capital, supply, prices, WTP, customers, beta, turns, visibility, production cap). |
| `tournament.py` | Match scheduling, parallel execution, official-style ranking. |
| `evolve.py` | Genetic algorithm with elitism, hall of fame and a final playoff. |
| `experiments/*.yaml` | Experiment definitions. |

## Commands (run from the repository root)

```bash
uv run python -m swarm params                                  # all knobs, defaults, ranges
uv run python -m swarm evolve -e swarm/experiments/quick.yaml  # ~5 s smoke run
uv run python -m swarm evolve -e swarm/experiments/full.yaml --export my-champion
uv run python -m swarm evolve -e swarm/experiments/full.yaml --set population=128 --set selection.elite=12

# Fixed strategies head to head: presets, saved results, bundles, random ones
uv run python -m swarm tournament --preset all --params outputs/swarm/<run>/playoff.json \
    --bundle src/dme_simulator/agents/test-agent --random 4 --rounds 50 --save ranking.json

uv run python -m swarm export --params outputs/swarm/<run>/champion.json --name my-champion
```

`--set key.sub=value` overrides any experiment key (values are parsed as YAML)
for both `evolve` and `tournament`.

## Scoring

Each match ranks teams by final cash. **Fitness** is the mean normalised rank:
0 means first in every match and 1 means last in every match. The leaderboard
also shows:

- the official-style mean rank;
- mean return (final cash / starting cash);
- win rate;
- the score per scenario, where `*` marks perturbed variants.

## Outputs

Each run writes `outputs/swarm/<name>-<timestamp>/`, containing:

- `experiment.yaml`, the resolved settings;
- `generation_NNN.json`, the ranked population with params and stats;
- `playoff.json`, the finalists ranked;
- `champion.json`.

Any of these JSON files can be passed to `--params` or to `init.seed_from` to
continue a search.

## Workflow

1. Run `full.yaml`, then inspect `playoff.json`. Check that the champion's score
   holds up on the perturbed `*` scenarios as well as the stock ones.
2. Seed the next search from it (`--set init.seed_from=.../playoff.json`).
   Optionally narrow `space:` or pin `fixed:` values.
3. Export the champion and check it with `dme-sim --profile`. Then upload the
   bundle folder, which contains only `agent.py` and `strategy.py`.
4. To add new behaviour, add a key to `DEFAULT_PARAMS`, use it in
   `ParamAgent`, and give it a range in `space.SPECS`. The module checks at
   import time that the two stay in sync.
