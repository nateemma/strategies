"""035 T013: NI_ signal-table registration adds names without touching existing ones."""

from __future__ import annotations

import json


def test_existing_table_plans_unchanged_and_ni_names(tmp_path, monkeypatch):
    from research.new_inputs import arms as A
    from research.nnnc_framework import signals as SG

    assert SG.PLANS["ARO"] == "wf_aro_b030" and SG.CO_PLANS["CO_ARO"] == "co_aro"
    assert SG.WR_PLANS["WR_ARO"] == "wr_aro_sz" and "LS_FMAE" in SG.LS_PLANS
    assert not set(SG.NI_PLANS) & set(SG.PLANS | SG.CO_PLANS | SG.WR_PLANS | SG.LS_PLANS)
    monkeypatch.setattr(A, "HERE", tmp_path)
    assert A.table_plans() == {}
    (tmp_path / "survivors.json").write_text(json.dumps({"survivors": ["PO_OI"]}))
    assert A.table_plans() == {"NI_PO_OI": "ni_po_oi"}
    assert A.class_name("PO_OI") == "NNNC_NI_PO_OI" and A.plan_tag("PO_OI", "sz") == "ni_po_oi_sz"


def test_gbr_names_and_unrecorded_reproduction(monkeypatch, tmp_path):
    from research.new_inputs import arms as A
    from research.new_inputs import trades as TR

    assert A.class_name("MS_VR", "GBR") == "NNNC_NI_GBR_MS_VR"
    assert A.plan_tag("MS_VR", "sz", "GBR") == "ni_gbr_ms_vr_sz"
    assert A.table_name("MS_VR", "GBR") == "NI_GBR_MS_VR" and A.table_name("MS_VR") == "NI_MS_VR"
    assert A.class_name("MS_VR") == "NNNC_NI_MS_VR"  # ARO names unchanged
    monkeypatch.setattr(TR, "PREREG_GBR", tmp_path / "none.json")
    assert TR.expected("GBR") is None and TR.expected("ARO") == (412, 0.1004)
