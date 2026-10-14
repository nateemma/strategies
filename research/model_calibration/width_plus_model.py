"""Does the model add on top of the band-width selection? (2026-10-13, offline diagnostic.)

On the width_twin causal selection: (a) AND the model's own entry (enter_long); (b) AND p_buy >= 0.6.

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.model_calibration.width_plus_model
"""

import dataclasses, json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from research.exit_coupling import feasibility as F
from research.nnnc_framework import entries as E
from research.model_calibration.width_twin import _gates
cuts_all = json.load(open("research/model_calibration/width_twin.json"))["models"]
out = {}
for m in ("ARO", "GBR", "GFM", "MLX"):
    model = E.Model(m)
    model.params = dataclasses.replace(model.params, stoploss_grace_hours=24.0)
    cuts = cuts_all[m]["cuts"]; folds = sorted(model.folds)
    res = {k: {"early": [0.0, 0], "late": [0.0, 0]} for k in ("width_and_model", "width_and_p0.6")}
    for i, f in enumerate(folds):
        t = model.tables[f]; buy = E.fold_params(model.prefix, f)
        g = _gates(t, buy); w = t["bb_width"].to_numpy(float)
        causal = cuts[folds[i - 1]] if i else cuts[f]
        wide = g & (w >= causal)
        for k, mk in (("width_and_model", wide & (t["enter_long"].to_numpy() == 1)),
                      ("width_and_p0.6", wide & (t["p_buy"].to_numpy() >= 0.6))):
            d = model.simulate(f, mk); h = F.half_of(f)
            res[k][h][0] += float(d["pp"].sum()) if not d.empty else 0.0; res[k][h][1] += len(d)
    out[m] = {k: {h: {"pp": round(v[0], 2), "trades": v[1]} for h, v in r.items()} for k, r in res.items()}
    print(m, json.dumps(out[m]), flush=True)
json.dump(out, open("research/model_calibration/width_plus_model.json", "w"), indent=1)
