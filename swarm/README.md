# Strategy swarm

Several families of parametrised agents, a parallel tournament runner, and an
island-model evolutionary search that picks the strongest parameter sets. The
package lives outside `src/dme_simulator/agents/`, so `dme-sim` discovers none
of it until you **export** a champion.

## Strategy families (`families/`)

Each family is a different idea of how to play, with its own parameters,
search space (`space.py`) and presets (`presets.py`).

| Family | Core idea |
| --- | --- |
| `adaptive` | Cost-plus price driven by sell-through; markups adapt to fill rate; price ladder; optional stockpiling when cheap. |
| `valuation` | Learns a demand curve per good: listed/sold stats per log-price bin, an optimistic prior, acceptance forced to fall as price rises, and softmax ladders. Material bids are capped by the value of the output. |
| `glut` | Times the auction: buys big batches only when the reserve has fallen well below its peak, and panic-buys when short. Sells in a premium tier and a value tier. |
| `rival` | Reads competitors from public state. Avoids and undercuts goods that rivals hold, charges a premium where nobody can supply, and raises bids when supplier lots get drained. |
| `neutral` | Never trades, so capital stays at its starting value. A zero-risk benchmark with no parameters: run it as a one-member island (`population: 1`, `selection: {elite: 1, immigrants: 0}`). |
| `portfolio` | Treats goods as bandit arms (UCB on realised return on capital, softmax budget split). PI controllers steer prices toward a target sell-through and markups toward a target fill rate. |
| `revman` | Revenue management (Gallego & van Ryzin; Gallego & Hu; Besbes & Zeevi). Learns a logistic demand curve per good and values stock by a finite-horizon DP under a prior on the unknown game length. Prices by `dp`, the static logit rule (`myopic`) or a `runout` price, and buys units whose marginal value beats the kit cost. |
| `auction` | Auction theory (first-price best response; Han, Zhou & Weissman; Ausubel et al.; Balseiro & Gur; Wellman et al.; RoxyBot-06). Learns clearing-price distributions from censored fills and bids kit-level best responses as a per-good ladder. Options: quantile or shade bidding, pacing, and SAA joint bids. |
| `capacity` | Capacity commitment (Kreps & Scheinkman; Kalymon / Golabi). Holds a restrained stock that sells within `cover` turns, optionally as a price-dependent material base stock, bought at a target win probability and sold at a run-out price. |
| `tactex` | Predict, then optimise (Pardoe & Stone; Kiekintveld et al.; Ketter et al.). Every engine module is a parameter, plus a scarcity/glut regime switch, so evolution chooses which strategies to combine. |

Family modules import only the public agent interface, plus `common.py` and
the sibling modules they list in `HELPERS`. Export copies `agent.py`, the
family module, `common.py` and those helpers into a bundle that the loader
accepts as-is.

### The paper-based engine (`families/econ.py`)

`revman`, `auction`, `capacity` and `tactex` share one engine. Each turn it
works backwards from sales to bids:

1. **Learn.** The chance that one listed unit sells is
   `sigmoid(a - b ln p + offset)`, fitted per good by online logistic
   regression with a prior. Per material it keeps the distribution of the
   clearing markup over the reserve, updated from full, partial and missed
   fills (or the whole log when it is visible).
2. **Value.** A unit on the shelf is worth the value of the selling policy:
   the finite-horizon DP, relisting at the static logit price, or the run-out
   price. A lognormal prior on the game length turns the unknown end into
   survival odds per turn.
3. **Buy.** The quantity comes from marginal value against kit cost, a
   stock cover, or a base stock. Bids come from a kit best response, a
   quantile or value shading, optionally with pacing and an SAA refinement.
   It stops buying near the expected end.
4. **Build** in value order, then **sell**, with one listing per unit
   (`listing: units`) or a few price rungs (`listing: rungs`).

Each family pins some modules in `FIXED` and evolves the rest. The
`ENGINE_SPECS` in `space.py` give the ranges for everything theory cannot pin
down: demand priors, forgetting, the horizon prior, risk attitude, margins and
learning rates. A per-decision time guard (`TIME_BUDGET`) skips optional work
under load, and any exception falls back to listing current stock.

**Per-unit listings.** The simulator weights listings, not teams: a customer
picks among affordable listings with weight exp(-beta x price) each. So ten
one-unit listings get ten times the pull of one listing of ten. The README
allows splitting a good across several listings, but this is a quirk worth
confirming with the organisers. Every engine family has a `rungs` preset for
when per-unit listings are not wanted (pin it with
`fixed: {listing: rungs}`).

To add a family: create `families/<name>.py` with `DEFAULTS` and `AGENT`,
register it in `families/__init__.py`, and add its ranges to `space.SPACES`.
Presets are optional. Import-time checks keep defaults and spaces in sync.

## Islands

`evolve` runs one island per family listed under `families:`. Each generation:

1. Every member of every island is shuffled into matches (`mixing: mixed`),
   together with fixed opponents (built-ins, bundles) and hall-of-fame members.
   With `mixing: separate`, families only meet themselves and the opponents.
2. All matches run on one process pool. Fitness is the mean normalised rank:
   0 means first every match, 1 means last every match.
3. Each island selects, crosses over and mutates **within itself**. Elite and
   immigrant counts are fractions of the island when below 1.
4. The best of each island joins a global hall of fame, used as opponents later.

At the end, a **playoff** puts the top `per_family` of each island, plus the
hall of fame, into the same matches. The run writes `playoff.json`,
`family_champions.json` (best per family) and `champion.json`.

## Commands (run from the repository root)

```bash
uv run python -m swarm params                    # every family, parameter, default and range
uv run python -m swarm params valuation glut

uv run python -m swarm evolve -e swarm/experiments/islands.yaml --export my-champion
uv run python -m swarm evolve -e swarm/experiments/islands.yaml \
    --set generations=60 --set families.glut.population=48 --set mixing=separate

# Head to head: presets (family:name, family:all, all), saved results, bundles, randoms
uv run python -m swarm tournament -e swarm/experiments/generated.yaml --preset all \
    --params outputs/swarm/<run>/family_champions.json --bundle src/dme_simulator/agents/test-agent \
    --random 10 --rounds 100 --save ranking.json

uv run python -m swarm export --params outputs/swarm/<run>/family_champions.json --index 2 --name glut-best
uv run python -m swarm export --preset rival:niche --name rival-niche
```

## Experiment files (`experiments/`)

| File | Scenarios | Families |
| --- | --- | --- |
| `quick.yaml` | default and scaled market, randomised | adaptive |
| `full.yaml` | the four supplied configs, randomised | adaptive |
| `generated.yaml` | `config/generated/*` | adaptive |
| `islands.yaml` | `config/generated/*` | all five, mixed |
| `islands_robust.yaml` | `config/generated/*`, match sizes 2–16 | all five, seeded from `islands` |
| `islands_neutral.yaml` | `config/generated/*`, match sizes 2–16 | all six including `neutral`, seeded from `islands_robust` |
| `papers.yaml` | `default.yaml` (about a fifth), `scaled_market`, `config/generated/*`, 40% perturbed; about 27–35 teams per match with a starter-agent crowd | `revman`, `auction`, `capacity`, `tactex` |
| `papers_robust.yaml` | as `papers`, plus 0–6 lot-draining whales per match; seeded from a `papers` run | the four engine families; `tactex` seeded with converted revman, auction and capacity champions |

**Match size matters.** Rankings change a lot between small and crowded
markets. In the first island run, `glut` won a playoff that put all 31 teams
in one match, but it came near the bottom of the evolved agents once matches
were small. Official match sizes are unknown, so give `match_size` (and
`playoff.match_size`, or `tournament --match-size 2 4 8 16`) a **list**. One
size is drawn per round, and strategies are selected for robustness across
field sizes.

Keys you will most often change:

- `scenarios`: files, folders or globs, with weights.
- `stratified`: deal scenarios evenly instead of by weight.
- `perturb`: procedural randomisation of scenarios (see `scenarios.py`); `null` turns it off.
- `families.<name>`: `population`, `presets`, `seed_from` (one or more result
  JSONs), `space`, `fixed` and per-island `selection` overrides.
- `opponents`, `selection` and `playoff`. `opponents.crowd` adds clones
  (`{bundle: ..., count: n}`, `{builtin: ..., count: n}`,
  `{preset: "family:name", count: n}` or
  `{id: ..., family: ..., params: {...}, count: n}`) to every match, so
  matches reach the portal's field size of about 30 teams. With
  `probability: p` a group joins only that share of the matches, which varies
  how harsh the field is. Leaderboards group clones as `label*`.
- `families.<name>.seed_families: {family: n}` seeds an island with the best
  `n` members of other families found in `seed_from`, converted into its own
  parameters (engine families share one vocabulary). This is how `tactex`
  starts from the revman, auction and capacity champions. A family module can
  narrow module choices with `CHOICES`; for example, `capacity` keeps only
  stock-target quantity rules.

Older single-family files (`population`, `init`, `space`, `fixed` at the top
level) still work and run one `adaptive` island. Defaults live in
`evolve.DEFAULT_EXPERIMENT`.

## Outputs

Each run writes `outputs/swarm/<name>-<timestamp>/`, containing:

- `experiment.yaml`, the resolved settings;
- `generation_NNN.json`, each island ranked, plus the hall of fame;
- `playoff.json`, `family_champions.json` and `champion.json`.

Every entry carries `family` and `params`. Any of these files can be passed to
`--params` or `seed_from`.
