# Contract: Arms as configurations of brief 01's trainer

The three arms are **configurations**, not implementations. Anything that differs between them
beyond initialisation and objective is a contract violation, because it destroys attribution.

## Shared and fixed across all arms

Simulator, cost model, feature pipeline, fold plan and universe, participation floor, held-out
ranking, trial log, seeds, and the deployable wrapper — all inherited from brief 01 unchanged
(FR-001). The encoder architecture is also shared: Arm A inherits its weights, B and C
initialise the same architecture randomly (FR-004a).

## What each arm may vary

| Arm | Init | Objective |
|---|---|---|
| A `warm_start` | classifier encoder, per fold | brief 01's utility loss; encoder LR ≥10× below head LR |
| B `blended` | random | `α·CE(label) + (1−α)·utility`, α ∈ {0.75, 0.5, 0.25}, both labels |
| C `auxiliary` | random | utility + auxiliary forward-volatility head; no directional label |

## Invariants

| Invariant | Requirement |
|---|---|
| Arms differ only in init and objective | FR-002 |
| Brief 01's modules are extended, never forked | FR-001 |
| A defect found in shared code is fixed in place, with a value-asserting test | FR-003 |
| Every arm with a forward-looking term purges and embargoes the horizon | FR-013, FR-014 |
| Each loss term's realised contribution is measured, not assumed | FR-009 |
| No directional label enters Arm C at any point | FR-011 |
| The blend sweep never reports a "best" weight | FR-010, SC-010 |
| Configurations are logged before they run | FR-015 |
| Ranking is on a held-out range | FR-017 |
| An arm below the participation floor is inconclusive, not a return | FR-020, SC-015 |

## Failure modes the contract requires to be explicit

- Inherited weights that do not differ from random initialisation → the warm-start silently did
  nothing; the run is void, not a null result.
- `alpha_realised` near 0 or 1 → the blend collapsed to one term; report it rather than the
  nominal weight.
- A non-finite loss → abort and record as abandoned (inherited from 009).
