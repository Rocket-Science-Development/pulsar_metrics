"""Aggregate data in FairnessAnalyzer (lab FA-03): row weights, composition tables, cluster bootstrap, numeric
attributes. Synthetic communities only."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.fairness import FairnessAnalyzer, FairnessPolicy
from pulsar_metrics.fairness.analyzer import AllCellsSuppressed

GROUPS = ("A", "B", "C")


def communities(n=600, seed=0):
    """Units with a population, the shares of three groups and a unit-level decision that rises with the B share"""
    rng = np.random.default_rng(seed)
    shares = rng.dirichlet([6.0, 1.0, 1.5], n)
    population = rng.lognormal(9, 1, n).round()
    flag = (rng.random(n) < 0.15 + 0.6 * shares[:, 1]).astype(int)
    y_true = (rng.random(n) < 0.2 + 0.5 * shares[:, 1]).astype(int)
    preds = pd.DataFrame({"cid": np.arange(n), "y_pred": flag, "y_true": y_true, "population": population})
    comp = pd.DataFrame([{"cid": i, "group": g, "share": shares[i, j]} for i in range(n) for j, g in enumerate(GROUPS)]).astype(
        {"share": float}
    )
    return preds, comp, shares


def exposure(preds, shares, j):
    """Population-weighted share of the people of group j living in a flagged unit"""
    w = preds["population"].to_numpy() * shares[:, j]
    return float((w * preds["y_pred"].to_numpy()).sum() / w.sum())


def get(results, name, group):
    return next(r for r in results if r.metric_name == name and r.group == group)


def test_row_weights_weight_every_group_rate():
    preds, _, shares = communities()
    frame = preds.assign(region=np.where(shares[:, 1] > 0.15, "high", "low"))
    policy = FairnessPolicy(criteria=["demographic_parity_difference", "equal_opportunity_difference"], n_bootstrap=200)
    results = FairnessAnalyzer(["region"], policy, reference_groups={"region": "low"}, weights="population").run(frame)
    w, f, y = frame["population"].to_numpy(), frame["y_pred"].to_numpy(), frame["y_true"].to_numpy()
    high = frame["region"].to_numpy() == "high"

    def rate(mask):
        return (w[mask] * f[mask]).sum() / w[mask].sum()

    dp = get(results, "demographic_parity_difference", "region=high")
    assert dp.metric_value == pytest.approx(rate(high) - rate(~high))
    eo = get(results, "equal_opportunity_difference", "region=high")
    assert eo.metric_value == pytest.approx(rate(high & (y == 1)) - rate(~high & (y == 1)))
    assert dp.n == int(high.sum())  # n counts rows (units), not people
    assert "weighted rates (population)" in dp.method and "Wilson" not in dp.method and dp.ci_low < dp.metric_value


def test_composition_table_gives_ecological_population_weighted_exposure():
    preds, comp, shares = communities()
    analyzer = FairnessAnalyzer(
        ["group"],
        FairnessPolicy(criteria=["demographic_parity_difference", "disparate_impact_ratio"], n_bootstrap=300),
        reference_groups={"group": "A"},
        weights="population",
        unit="community",
    )
    results = analyzer.run(preds, attributes_data=comp, on="cid", share="share")
    dp = get(results, "demographic_parity_difference", "group=B")
    assert dp.metric_value == pytest.approx(exposure(preds, shares, 1) - exposure(preds, shares, 0))
    ratio = get(results, "disparate_impact_ratio", "group=C")
    assert ratio.metric_value == pytest.approx(exposure(preds, shares, 2) / exposure(preds, shares, 0))
    assert dp.status == "ecological" and dp.n == len(preds)
    assert "cluster bootstrap over 600 community units" in dp.method and "ecological" in dp.method
    assert dp.ci_low < dp.metric_value < dp.ci_high


def test_count_composition_needs_no_weights():
    preds, comp, shares = communities()
    counts = comp.assign(residents=comp["share"] * preds["population"].to_numpy()[comp["cid"].to_numpy()])
    policy = FairnessPolicy(criteria=["demographic_parity_difference"], n_bootstrap=100)
    results = FairnessAnalyzer(["group"], policy, reference_groups={"group": "A"}).run(
        preds, attributes_data=counts[["cid", "group", "residents"]], on="cid", count="residents"
    )
    expected = exposure(preds, shares, 1) - exposure(preds, shares, 0)
    assert get(results, "demographic_parity_difference", "group=B").metric_value == pytest.approx(expected)


def expanded(preds, shares, copies):
    """Pseudo-individual rows: `copies` rows per unit and group, in proportion to the shares (rounded)"""
    parts = []
    for j, g in enumerate(GROUPS):
        k = np.round(copies * len(GROUPS) * shares[:, j]).astype(int)
        parts.append(
            pd.DataFrame(
                {"group": g, "y_pred": np.repeat(preds["y_pred"].to_numpy(), k), "cid": np.repeat(preds["cid"].to_numpy(), k)}
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_cluster_bootstrap_does_not_shrink_with_the_number_of_copies():
    preds, _, shares = communities()
    policy = FairnessPolicy(criteria=["demographic_parity_difference"], n_bootstrap=300)
    widths = {}
    for copies in (10, 100):
        frame = expanded(preds, shares, copies)
        for cluster in (None, "cid"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                r = FairnessAnalyzer(["group"], policy, reference_groups={"group": "A"}, cluster=cluster).run(frame)
            dp = get(r, "demographic_parity_difference", "group=B")
            widths[copies, cluster] = dp.ci_high - dp.ci_low
            copy_warning = any("repeat another row" in str(w.message) for w in caught)
            assert copy_warning is (cluster is None)
            if cluster:
                assert dp.n <= len(preds) and "cluster bootstrap" in dp.method
    # rows as independent people: the width shrinks as 1/sqrt(copies); units resampled: it does not move
    assert widths[100, None] < 0.45 * widths[10, None]
    assert 0.75 < widths[100, "cid"] / widths[10, "cid"] < 1.33
    assert widths[100, "cid"] > 2 * widths[100, None]


def test_composition_without_share_is_refused_with_an_explicit_message():
    preds, comp, _ = communities(n=50)
    with pytest.raises(error_msg, match="share= or count="):
        FairnessAnalyzer(["group"], FairnessPolicy(n_bootstrap=0)).run(preds, attributes_data=comp, on="cid")
    with pytest.raises(error_msg, match="one prediction per id"):
        FairnessAnalyzer(["group"], FairnessPolicy(n_bootstrap=0)).run(
            pd.concat([preds, preds]), attributes_data=comp, on="cid", share="share"
        )


def test_numeric_attribute_hints_binning_and_bins_by_quantile():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"share": rng.beta(0.5, 8, 2000), "y_pred": rng.integers(0, 2, 2000)})
    with pytest.warns(UserWarning, match="numeric with 2000 distinct values"):
        with pytest.raises(AllCellsSuppressed, match=r"bins=\{'share': 3\}"):
            FairnessAnalyzer(["share"], FairnessPolicy(n_bootstrap=0)).run(frame)
    analyzer = FairnessAnalyzer(["share"], FairnessPolicy(n_bootstrap=100), bins={"share": 3}, reference_groups={"share": "Q1"})
    results = analyzer.run(frame)
    assert {r.group.split(" ")[0] for r in results} == {"share=Q2", "share=Q3"}
    assert results[0].reference_group.startswith("share=Q1 [") and results[0].n in (666, 667)
    assert len(analyzer.bin_edges["share"]) == 4 and "share binned, edges" in results[0].method
    edges = FairnessAnalyzer(["share"], FairnessPolicy(n_bootstrap=0), bins={"share": [0, 0.05, 1]}).run(frame)
    assert {r.group.split(" ")[0] for r in edges} | {edges[0].reference_group.split(" ")[0]} == {"share=B1", "share=B2"}
