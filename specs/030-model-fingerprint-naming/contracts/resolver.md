# Contract: model-key resolver and CLI (030)

## Python (`Framework/ModelKey.py`, generic: it reads only the declared lists, holds no family names)

- `resolve(strategy) -> ModelKey`
  - Works on a live strategy instance; needs `dp` for the whitelist.
  - Pure apart from file hashing, which is cached.
  - Raises `ModelKeyError` naming the missing attribute or unreadable file.
- `key_hash(key) -> str` (16 hex characters), `key_prefix(key) -> str`.
- `locate(strategy) -> Location(path, kind, hash)`.
  - Follows the lookup order of research R8: override, then index, then legacy, then new.
  - Never trains.
- `check_manifest(location, key)`: raises `ModelKeyMismatch(fields)` (FR-008).
- `record_training(location, key, actual_timerange)`: writes the manifest and index entry after a
  successful train (FR-010).
- Report mode: `FT_MODEL_KEY=report`. It resolves and logs the key and the manifest comparison but
  returns the pre-030 path (R11). It is used once before strict switch-on.
- Arm records gain `model_key` (`hash`, `prefix`, `kind`), written by `regime/run_arm.py` (R12).
- Strict mode: `FT_MODEL_EVAL_STRICT=1`.
  - In strict mode, `locate` on a missing model raises `ModelMissing(hash, prefix, key)` before any
    training starts (FR-009).

## Hook in Framework (generic)

- `BaseNNStrategy.get_model_path()` calls `ModelKey.locate(self).path`.
  - Class-level overrides keep winning by normal method resolution.
  - Family-specific inputs enter only through the families' own `MODEL_KEY_*` declarations.
- `TrainingEngine.maybe_train`, after a successful `train_model`, calls `record_training`.

## CLI (`scripts/model_index.py`)

| command | effect | exit |
|---|---|---|
| `key <Class> [--fold fNN --plan P]` | prints prefix, hash, the canonical key, resolved location and whether weights exist (FR-007) | 0 found, 1 missing |
| `path <Class>` | prints the resolved weight path (for `test_strat.sh -f`) | 0 |
| `migrate [--dry-run]` | indexes the walk-forward sets from plans and single dirs as legacy; writes manifests; reports conflicts and legacy entries | 0 clean, 3 conflicts |
| `audit` | runs the R5 classification and the legacy-redirect register check outside pytest | 0 / 1 |
| `verify-bytes <snapshot.json>` | compares every weight file's sha256 with a snapshot (FR-020) | 0 / 1 |
