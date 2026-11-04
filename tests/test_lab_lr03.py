"""Regression tests for the limits found by lab test LR-03 (RAG agent on a real corpus, drift trigger)."""

import numpy as np
import pytest

from pulsar_metrics.exceptions import CustomExceptionPulsarMetric as error_msg
from pulsar_metrics.genai import embedding_drift, mmd_test


def _runs(rng, n_runs, passages=2, d=8, noise=0.1):
    """Embeddings of the passages retrieved by n_runs runs: the passages of a run sit close to their query's topic"""
    topics = rng.normal(size=(n_runs, d))
    rows = np.repeat(topics, passages, axis=0) + rng.normal(scale=noise, size=(n_runs * passages, d))
    return rows, np.repeat(np.arange(n_runs), passages)


def test_mmd_groups_keep_the_level_when_the_passages_of_a_run_are_correlated():
    """LR-03: rows permuted as independent gave 42-44 % A/A false alarms on 2 passages per run; whole runs permuted
    hold the 5 % level"""
    rng = np.random.default_rng(0)
    rows, grouped = [], []
    for s in range(40):
        (x, gx), (y, gy) = _runs(rng, 60), _runs(rng, 30)
        rows.append(mmd_test(x, y, n_permutations=100, n_components=None, seed=s).p_value < 0.05)
        grouped.append(mmd_test(x, y, n_permutations=100, n_components=None, seed=s, groups=(gx, gy)).p_value < 0.05)
    assert sum(rows) >= 20  # the row permutation null is far too narrow
    assert sum(grouped) <= 5  # 40 A/A draws, 5 % nominal


def test_mmd_groups_detect_a_change_and_pass_through_embedding_drift():
    rng = np.random.default_rng(1)
    (x, gx), (y, gy) = _runs(rng, 60), _runs(rng, 30)
    y = y + 1.0  # every passage moves
    result = embedding_drift(x, y, n_permutations=200, groups=(gx, gy))
    assert result.drift_status is True and result.n == 60


def test_mmd_singleton_groups_match_the_row_test():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(50, 6)), rng.normal(size=(40, 6)) + 0.3
    plain = mmd_test(x, y, n_permutations=200)
    singletons = mmd_test(x, y, n_permutations=200, groups=(np.arange(50), np.arange(40)))
    assert singletons.p_value == plain.p_value and singletons.statistic == plain.statistic


def test_mmd_groups_subsample_whole_groups_and_check_their_shape():
    rng = np.random.default_rng(3)
    (x, gx), (y, gy) = _runs(rng, 30, passages=3), _runs(rng, 30, passages=3)
    result = mmd_test(x, y, n_permutations=50, max_samples=40, groups=(gx, gy))
    assert result.n_reference == 39 and result.n_current == 39  # 13 whole runs of 3 passages
    with pytest.raises(error_msg):
        mmd_test(x, y, groups=(gx[:-1], gy))
    with pytest.raises(error_msg):
        mmd_test(x, y, groups=gx)
    with pytest.raises(error_msg):
        mmd_test(x, y, groups=(np.zeros(len(x)), gy))  # a single group in the reference


def test_mmd_group_subsampling_keeps_two_groups_when_one_exceeds_max_samples():
    """Groups larger than max_samples: two whole groups are kept, not one (which made the test fail)"""
    rng = np.random.default_rng(4)
    (x, gx), (y, gy) = _runs(rng, 4, passages=30), _runs(rng, 4, passages=30)
    result = mmd_test(x, y, n_permutations=20, max_samples=20, groups=(gx, gy))
    assert result.n_reference == 60 and result.n_current == 60
    with pytest.raises(error_msg, match="reference holds 1"):
        mmd_test(x, y, groups=(np.zeros(len(x)), gy))
