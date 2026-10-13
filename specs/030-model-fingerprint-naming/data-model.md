# Data Model: Model Naming by Model Fingerprint (030)

## ModelKey

The resolved inputs, as canonical JSON with sorted keys and no whitespace.

| field | content |
|---|---|
| `version` | key schema version, 1 |
| `code_version` | `MODEL_CODE_VERSION`, starting at 1, bumped by hand when training behaviour changes |
| `family` | the in-scope family root, e.g. `NNNCStrategy` |
| `attributes` | `{name: value}` for every `MODEL_KEY_ATTRIBUTES` member, resolved on the instance after `.json` params merge. Values are made JSON-safe: enums by name, sets sorted, tuples become lists. |
| `methods` | `{method: "module.QualName"}` for the class that defines each training-path method |
| `classifier` | classifier class qualname, plus its training knobs (`max_epochs`, `early_patience`, `weight_decay`, `learning_rate`) and the instance attributes mixins set on it |
| `files` | `{role: sha16 \| "fitted-at-train:<name>" \| "absent"}`, where role is one of `main_scaler`, `pca`, `tensor_scaler`, `gan_generator`, `gan_scaler_a`, `coeff_*` |
| `models` | `{entropy_guidance: <key hash or legacy key>}` when used |
| `run` | `train_timerange` (or `"unpinned"`), `whitelist` (ordered list), `timeframe`, `startup_candle_count`, `seed` |
| `trained_with` | the `MODEL_TRAINED_WITH` pins applied (already reflected in `attributes` and `run`), so they are visible |

**hash:** SHA-256 of the canonical JSON, first 16 hex characters.

**prefix:** `<classifier_type>_<label>_g<gain>_h<horizon>`, lower-case and character-safe, e.g.
`lstm_gbbpath_g0.02_h48`. It is never used for identity.

**Validation:** resolution fails loudly if an attribute is missing, or a file role is required
but unreadable.

## Manifest (`manifest.json` beside the weights)

| field | content |
|---|---|
| `key` | ModelKey |
| `hash`, `prefix` | |
| `kind` | `keyed` (trained under 030), `migrated` (walk-forward set indexed from a plan) or `legacy` (single model, per-class key) |
| `provenance` | `trained` (with the run's actual timerange); `reconstructed-from-plan` (plan path, run_id, params_sha, fold); or `legacy-unknown` (with a list of unknown inputs) |
| `weights` | relative path, size, sha256 of the bytes at index time |
| `users` | class names seen resolving to this key, each with its `trained_with` pins. Append-only. |
| `written` | ISO time |

**Rules:**
- `kind=legacy` resolves only for its own class name.
- A load whose resolved key doesn't match the manifest's key halts with the differing fields
  (FR-008). Legacy models are exempt, because they carry no full key.

## Model index (`saved_data/model_index.json`)

`{hash | "legacy:<Class>": {path, kind, manifest}}`

- One entry per key.
- **Conflicts:** `{"conflicts": [{hash, paths, reason}]}`. A hash in conflict is never served.
- **Snapshot:** a copy is committed as `specs/030-model-fingerprint-naming/model_index_migrated.json`
  at migration time.

## Legacy redirect register (`specs/030-model-fingerprint-naming/legacy_redirects.json`)

`{class: {defined_in, target, reason}}` for every class whose effective `get_model_path` is
a class-level override (R8). A test keeps this register equal to the AST scan, so a new override
can't appear unregistered.

## Declarations on strategy classes

| attribute | meaning |
|---|---|
| `MODEL_KEY_ATTRIBUTES` | tuple, merged along the MRO: the generic Framework list plus family extensions |
| `MODEL_KEY_METHODS` | tuple, merged: training-path method names |
| `MODEL_KEY_IGNORED` | tuple, merged: classified post-model names (the R5 audit) |
| `MODEL_TRAIN_TIMERANGE` | `"YYYYMMDD-YYYYMMDD"` or None; written into generated fold classes |
| `MODEL_SEED` | int or None; written into generated fold classes |
| `MODEL_TRAINED_WITH` | dict of pins naming training-time values that differ at run time (R2), e.g. `{"startup_candle_count": 64}` |
| `MODEL_CODE_VERSION` | int, Framework default 1 |

## State transitions (one key)

```
absent --train (non-strict run)--> keyed (manifest written)
absent --strict run--> HALT (names the key; no training)
legacy dir --migrate--> legacy (manifest, legacy key)
fold dir --migrate--> migrated (manifest, key from plan)
two dirs, one key, different bytes --migrate--> conflict (neither served)
```
