"""Convert a trained MLX model to its TensorFlow twin's weights (NNNC/NNNClassifierTF.py, NNMT/NNMTClassifierTF.py).

For each fold of a plan (or one --timerange, or a mainline class): locate the MLX class's weights by model
key, build the twin's Keras model, copy the weights, CHECK that both models give the same softmax on a
random batch (max |diff| <= --tol), and save the twin's .keras at the twin's own model key, with a manifest
recording the source key and weights. Never trains; refuses if the MLX weights are missing, the twin
already has weights (unless --force), or the check fails. Needs MLX (run it on the Mac that trained the
MLX models); the converted weights then load on any host with TensorFlow.

    cd user_data/strategies && ../../.venv/bin/python scripts/convert_mlx_to_tf.py \
        --mlx NNNC_CS_ARO --tf NNNC_CS_ARO_TF --plan evaluation/folds/plan_cs_aro.json [--folds f00,f14] [--seed 2]
    ... --mlx NNMT_LR_RI --tf NNMT_LR_RI_TF --config config/config_wf.json --timerange 20210101-20230101
    ... --mlx NNNC_CS_ARO --tf NNNC_CS_ARO_TF --config config/config_live_nnnc.json --mainline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STRAT = Path(__file__).resolve().parents[1]
ROOT = STRAT.parents[1]
sys.path[:0] = [str(STRAT), str(STRAT / "NNNC"), str(STRAT / "NNMT"), str(STRAT / "evaluation")]


def _pinned(cls, timerange: str | None, seed: int, name: str):
    if timerange is None:
        return cls
    return type(name, (cls,), {"MODEL_TRAIN_TIMERANGE": timerange, "MODEL_SEED": int(seed), "__module__": cls.__module__})


def convert_one(mlx_cls, tf_cls, config: dict, timerange: str | None, seed: int, tol: float, force: bool) -> dict:
    import mlx.core as mx

    from Framework import ModelKey as MK
    from Predictors.TFPortBasePredictor import read_safetensors

    src_cls = _pinned(mlx_cls, timerange, seed, f"{mlx_cls.__name__}_conv")
    dst_cls = _pinned(tf_cls, timerange, seed, f"{tf_cls.__name__}_conv")
    src = MK.locate_for(src_cls, config, src_cls.__name__)
    dst = MK.locate_for(dst_cls, config, dst_cls.__name__)
    src_w = Path(src.path).with_suffix(".safetensors")
    if src.kind in ("absent", "mainline-absent") or not src_w.is_file():
        raise SystemExit(f"REFUSED: no MLX weights for {mlx_cls.__name__} {timerange or 'mainline'} seed {seed} ({src.path})")
    dst_path = Path(dst.path)  # ".keras": the TF port saves the whole model there
    if dst.kind not in ("absent", "mainline-absent") and not force:
        raise SystemExit(f"REFUSED: {tf_cls.__name__} already has weights at {dst_path} (--force to overwrite)")

    seq_len = int(MK.declared(mlx_cls, "seq_len"))
    w = read_safetensors(src_w)
    num_features = int(w["lstm.Wx"].shape[1])
    strat = mlx_cls.__new__(mlx_cls)  # no __init__: only the classifier factories are needed
    mlx_clf = mlx_cls.get_classifier(strat, mlx_cls.classifier_type, "BTC/USDT", seq_len, num_features)
    mlx_model = mlx_clf.create_model(seq_len, num_features)
    mlx_model.load_weights(str(src_w))
    mlx_model.eval()
    tf_clf = tf_cls.get_classifier(strat, tf_cls.classifier_type, "BTC/USDT", seq_len, num_features)
    tf_model = tf_clf.create_model(seq_len, num_features)
    tf_clf.load_mlx_weights(tf_model, src_w)

    X = np.random.default_rng(0).normal(size=(4096, seq_len, num_features)).astype(np.float32)
    a, b = mlx_model(mx.array(X)), tf_model(X, training=False)
    if isinstance(a, dict):
        diff = max(float(np.abs(np.array(a[k]) - np.asarray(b[k])).max()) for k in a)
    else:
        diff = float(np.abs(np.array(a) - np.asarray(b)).max())
    if not diff <= tol:
        raise SystemExit(f"REFUSED: converted model differs from the MLX model (max |diff| {diff:.2e} > {tol:.0e})")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tf_model.save(dst_path)
    stem = dst_path.with_suffix("")
    kind = "mainline" if dst.kind.startswith("mainline") else "keyed"
    prov = {"kind": "converted_from_mlx", "timerange": timerange, "class": tf_cls.__name__,
            "source": {"class": mlx_cls.__name__, "hash": src.hash, "path": str(src_w.relative_to(STRAT)),
                       "sha": MK.file_sha(src_w)}, "max_abs_diff": diff}
    MK.write_manifest(stem.parent, dst.key, kind, prov, {dst_path.name: MK.file_sha(dst_path)}, [tf_cls.__name__])
    if kind == "keyed":
        storage = dst_path.parents[2]
        idx = MK.load_index(storage)
        idx["entries"][dst.hash] = {"stem": str(stem.relative_to(storage)), "kind": "keyed"}
        MK.save_index(storage, idx)
    return {"timerange": timerange, "seed": seed, "source": str(src_w.relative_to(STRAT)),
            "target": str(dst_path.relative_to(STRAT)), "max_abs_diff": diff}


def main(argv=None) -> int:
    from evaluation.fingerprint import load_config, load_strategy_class

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlx", required=True, help="the MLX strategy class (e.g. NNNC_CS_ARO)")
    ap.add_argument("--tf", required=True, help="its TensorFlow twin (e.g. NNNC_CS_ARO_TF)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", help="walk-forward plan: convert each fold (its config and train windows)")
    g.add_argument("--timerange", help="one training window, e.g. 20210101-20230101 (needs --config)")
    g.add_argument("--mainline", action="store_true", help="the mainline (no pinned window) model (needs --config)")
    ap.add_argument("--config", help="freqtrade config the MLX model was trained under (default: the plan's)")
    ap.add_argument("--folds", help="comma-separated fold ids (with --plan); default all")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    tf_cls = load_strategy_class(a.tf)
    mlx_cls = next((c for c in tf_cls.__mro__ if c.__name__ == a.mlx), None)  # the twin's own parent class
    if mlx_cls is None:
        raise SystemExit(f"REFUSED: {a.tf} is not a subclass of {a.mlx} (a twin must change only the model's framework)")
    if a.plan:
        plan = json.loads(Path(a.plan).read_text())
        cfg = a.config or plan["config"]
        want = set(a.folds.split(",")) if a.folds else None
        ranges = [f["train_timerange"] for f in plan["folds"] if want is None or f["fold_id"] in want]
    else:
        if not a.config:
            raise SystemExit("--config is required with --timerange / --mainline")
        cfg, ranges = a.config, [a.timerange if a.timerange else None]
    cfg_path = Path(cfg) if Path(cfg).is_absolute() else (ROOT / cfg if (ROOT / cfg).exists() else STRAT / cfg)
    config = load_config(cfg_path)
    out = [convert_one(mlx_cls, tf_cls, config, tr, a.seed, a.tol, a.force) for tr in ranges]
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
