"""Metric semantics and compatibility; no telemetry generation or optical run."""

import json
from pathlib import Path

import numpy as np
from noether_link_audit.attribution import LossModel, measurement_contract, shapley
from noether_link_audit.pipeline import _attribution_dict
from noether_link_audit.report import _findings, _markdown


def _model(cause: str) -> LossModel:
    return LossModel(
        rate_gbps=4.0,
        sample_s=0.25,
        sensitivity_dbm=-40.0,
        base_dbm=np.array([-41.0, -41.0, -40.0, -39.0], dtype=np.float32),
        locked=np.array([True, True, True, False]),
        additive={cause: np.array([2.0, 2.0, 0.0, 0.0], dtype=np.float32)},
    )


def test_threshold_proxy_keeps_legacy_numeric_serialization() -> None:
    # Equality at sensitivity counts; unlocked power above sensitivity does not.
    result = shapley(_model("a"), ("a", "b"))
    serialized = _attribution_dict(result)
    assert serialized == {
        "possible_gb": 4.0,
        "delivered_gb": 1.0,
        "threshold_delivered_gb": 1.0,
        "all_causes_removed_gb": 3.0,
        "link_budget_residual_gb": 1.0,
        "shapley_gb": {"a": 2.0, "b": 0.0},
    }
    contract = measurement_contract()
    assert contract["data_metric"] == "locked_power_threshold_proxy_gb"
    assert contract["measured_modem_goodput"] is False
    assert contract["causal_identification"] is False


def test_identical_observations_do_not_identify_shapley_causes() -> None:
    # Two assumed mechanisms fit identical power and lock observations. Exact
    # Shapley accounting cannot decide which mechanism produced those signals.
    first, second = _model("a"), _model("b")
    np.testing.assert_array_equal(first.base_dbm, second.base_dbm)
    np.testing.assert_array_equal(first.locked, second.locked)
    a, b = shapley(first, ("a", "b")), shapley(second, ("a", "b"))
    assert a.delivered_gb == b.delivered_gb == 1.0
    assert a.all_removed_gb == b.all_removed_gb == 3.0
    assert a.shapley_gb == {"a": 2.0, "b": 0.0}
    assert b.shapley_gb == {"a": 0.0, "b": 2.0}


def test_report_labels_legacy_results_without_rewriting_metrics() -> None:
    retained = Path(__file__).resolve().parents[1] / "results/link-audit-sample-v1/metrics.json"
    raw = retained.read_bytes()
    payload = json.loads(raw)
    assert "measurement_contract" not in payload  # Preserve old-format support.
    before = json.dumps(payload, sort_keys=True)
    markdown = _markdown(payload, _findings(payload), None)
    assert "power-threshold data proxy" in markdown
    assert "not measured modem goodput" in markdown
    assert "Shapley values do not establish physical causes" in markdown
    assert "not a proven upper bound" in markdown
    assert "The station delivered" not in markdown
    assert "Five fixes recover" not in markdown
    assert "Each lost gigabit is attributed to a cause" not in markdown
    assert json.dumps(payload, sort_keys=True) == before
    assert retained.read_bytes() == raw
