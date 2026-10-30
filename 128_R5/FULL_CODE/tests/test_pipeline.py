import numpy as np
import pandas as pd
from storepulse.pipeline import enforce_policy, emerging_flags


def test_policy_floor_and_non_issue_actionability():
    tax = {
        "IT18": {"policy_severity_floor": "high", "is_actionable_category": 1},
        "NI01": {"policy_severity_floor": "low", "is_actionable_category": 0},
    }
    _, severity, action = enforce_policy(
        np.array(["IT18", "NI01"], dtype=object),
        np.array(["low", "low"], dtype=object),
        np.array([1, 1]),
        pd.Series(["allergen label is missing at this store", "thank you very much for the help"]),
        pd.Series(["ST1", "ST1"]), tax,
    )
    assert severity[0] in {"high", "critical"}
    assert action.tolist() == [1, 0]


def test_emerging_flags_schema_and_evidence():
    fb = pd.DataFrame([
        {"feedback_id": "a", "created_at": "2026-04-06T10:00:00", "store_id": "S1", "text": "freezer broken"},
        {"feedback_id": "b", "created_at": "2026-04-07T10:00:00", "store_id": "S1", "text": "freezer not cold"},
        {"feedback_id": "c", "created_at": "2026-04-08T10:00:00", "store_id": "S1", "text": "freezer not working"},
    ])
    pred = pd.DataFrame({"feedback_id": ["a", "b", "c"], "issue_type_id": ["IT10"] * 3})
    stores = pd.DataFrame({"store_id": ["S1"], "store_name": ["Saffron Market Demo"], "avg_weekly_footfall": [1000]})
    out = emerging_flags(fb, pred, stores, as_of="2026-04-06")
    assert list(out.columns) == ["as_of_week_start", "store_id", "issue_type_id", "confidence", "evidence_feedback_ids"]
    assert len(out) <= 3
    if len(out):
        assert len(set(out.iloc[0].evidence_feedback_ids.split(";"))) >= 2
