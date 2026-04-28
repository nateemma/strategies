# reference — External Strategies

Strategies copied from other authors and kept here for reference.  None
of these are actively maintained or used.  Their purpose is:

* Cut-and-paste source for indicator combinations and entry/exit logic.
* Comparison baselines — does my strategy beat NostalgiaForInfinityX
  on the same timerange?
* Learning aid — how do other people structure their strategies?

These files do **not** follow this repo's conventions.  They generally
inherit directly from `IStrategy` rather than `BaseStrategy`, don't use
the shared `DataframePopulator`, and have their own indicator naming.
Treat them as untouchable; if you want to adopt a technique, copy and
re-shape it into your own strategy in the appropriate family directory.

## Main files

| File | Author / source |
|---|---|
| `NostalgiaForInfinityX.py` | iterativv — the canonical "kitchen sink" multi-strategy.  Hundreds of indicator combinations and entry/exit conditions. |
| `MacheteV8b.py` | Indicator-driven strategy emphasising mean-reversion entries. |
| `CryptoFrog.py` | froggleston — long-running BBRSI-style strategy. |

See the top-level `README.md` for the full list of reference repositories.
