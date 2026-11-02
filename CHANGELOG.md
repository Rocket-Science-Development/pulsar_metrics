# Changelog

All notable changes to pulsar-metrics. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - calibration-analyzer

Limits confirmed by the night lab of 2 November 2026 (C1, C2, M1 to M5, m1 to m3, m16, m21).

### Added
- `AACalibrator`: `fdr_alpha` (0.05), `auto_draws` (True) and `max_draws` (10000). `fit(n_tests=...)` raises the number
  of A/A draws until the empirical p-value floor `1 / (n_draws + 1)` is half of `fdr_alpha / m`; `n_draws_used` gives
  the number used. With `auto_draws=False`, a warning is emitted instead (C1).
- `AACalibrator.empirical_pvalues()` and `fdr_correct(..., pvalue_floor=)` warn when the floor is above `alpha / m`;
  `AACalibrator.pvalue_floor` (C1).
- `AACalibrator.errors`: a (metric, feature) that cannot be calibrated is recorded with a warning, and the others are
  calibrated (M3).
- `NullDistribution.status` (`ok`, `constant_reference`, `degenerate_null`), with a warning (M4, m2, m3).
- `NullDistribution.threshold_interval()`, `n_effective`, `reliable`; `AACalibrator(shrink_draws=, reliable_windows=30)`;
  `AACalibrator.summary()` (M5).
- `AACalibrator.options()` adds `"reliable"` to each entry, copied by the drift metrics to the new `reliable` field of
  their results; `options(include_reliability=False)` keeps the old shape (m1).
- `CalibrationCache` (in memory or `.npy` files) and `fit(cache=)`: null distributions keyed by the metric, a hash of the
  feature's reference values and the draw settings, reused across windows, runs and models. `AACalibrator.save()` /
  `AACalibrator.load()` (JSON) (m21).
- `DriftMetric` / `DriftTestMetric`: `missing_values` (sentinels turned into NaN) and `nan_policy` (`omit`, `propagate`,
  `raise`), also in `Analyzer.add_drift_metrics(missing_values=..., nan_policy=...)`, with a dict per feature (M1, M2).
- `missing_rate` drift test: Fisher exact test of the share of missing values (M1).
- `PerformanceMetric` / `Analyzer.add_performance_metrics`: `min_label_coverage`, `min_labelled` and `missing_values`
  for `y_true` / `y_pred` (C2, M1).
- `MetricResults`: `status`, `missing_share_current`, `missing_share_reference`, `n_labelled`, `label_coverage`,
  `reliable` (optional).

### Changed
- **Performance metrics are evaluated on the rows with ground truth** instead of failing as soon as one `y_true` is
  missing, and report `n_labelled` and `label_coverage`. Below `min_label_coverage` (default 0.0) the result carries an
  `error`. The old default was the defect: a window with 18 % of labels missing gave no performance at all (C2).
- **Drift metrics set NaN aside by default** (`nan_policy="omit"`) and record their share. A NaN statistic now gives
  `drift_status=None` and an `error` instead of `True` (distances) or `False` (tests). The old behaviour was the
  defect: one NaN made every verdict arbitrary (M2).
- **`AACalibrator.fit` refuses a reference shorter than 10 windows** unless `allow_short=True` (it used to warn and
  return thresholds used as reliable). Contiguous draws need 4 windows even with `allow_short`. A warning is emitted
  below `reliable_windows` (30) windows (m1, M5).
- **Default number of A/A draws**: 200 is now a minimum, raised with the number of tests (C1). Calibration takes
  longer with many features; cache it (`CalibrationCache`, `save` / `load`).
- Contiguous A/A draws are sampled uniformly over pairs of disjoint windows without rejection loop (m3): calibrated
  values differ slightly from the previous version for the same seed.
- A/A draws on numeric features run the statistic on numpy arrays (same values; the calibration is no slower despite
  the new missing-value handling).
- `Analyzer.run` compares `model_id` / `model_version` as text, and a float column holding integral values also without
  its ".0", so that a reference read back from a CSV file (`"1.0"` read as `1.0`, `"1"` read as `1.0` next to a
  missing value) is accepted. Nothing else is compared numerically: `"1.10"` and `"1.1"`, or `"007"` and `7`, stay
  distinct versions. The error says how many rows each filter kept (m16).
- **`psi`, `kl` and `chi2` bin numeric data as float whatever their dtype.** `pd.cut` labelled the bins of integer data
  differently and merged neighbouring bins (values 0 and 1 of a 0-4 feature fell in one bin as int64, in two as
  float64), so the value depended on the dtype: a column that became float in some windows only (masked sentinel) was
  binned differently from its reference, and the A/A calibration (float arrays) differed from production (psi
  threshold 0.0988 vs 0.0859, chi2 alpha 0.0478 vs 0.0367 on a 0-4 integer feature, i.e. 7 % A/A false alarms
  instead of 5 %). **Values change for integer features**; float features are unchanged (the reference file of the
  characterization tests still matches).
- Performance metrics evaluated on part of the window say so in `method` ("labelled rows only: n of N (coverage x %)").
  With `min_label_coverage=0` (default), a UserWarning is emitted below 50 % coverage. The default stays 0: any
  positive default would silently withhold the performance of windows that the old code refused outright, while the
  coverage is now reported in every result and the warning flags the thin cases; set `min_label_coverage` to refuse
  them (C2).
- An automatic increase of the A/A draws above 5 times `n_draws` is logged as a WARNING (logger
  `pulsar_metrics.calibration`) with the number of evaluations, an estimated cost and a pointer to `CalibrationCache`;
  smaller increases stay at INFO (C1).
- `Analyzer.run`: declared sentinels are also turned into NaN for the summary statistics (M1).
- Numeric series of different dtypes (int and float) are compared as float by the drift metrics.

## [0.10.0] - 2026-11-02

### Added
- `pulsar_metrics.genai.mmd_test` / `embedding_drift`: MMD permutation test on embeddings (Gaussian mixture kernel,
  widths 0.5 to 4 times the median distance; PCA fitted on both windows pooled, without labels, so that the
  permutation test stays valid).
- `pulsar_metrics.genai.model_equality_test` / `model_equality`: model equality on a canary set (Gao et al., ICLR 2025),
  MMD with a Hamming kernel within prompts, permutation p-value within prompts.
- `pulsar_metrics.genai.ComplianceMonitor`: compliance of agent trajectories with rules in LTLf or Declare templates
  (`existence`, `absence`, `init`, `precedence`, `response`, `chain_response`, `not_succession`, `allowed_tools`), loaded
  from YAML or dicts; violations per run, compliance rate per rule with a Wilson interval (metric type `agents`).
- `MetricResults.p_value` (optional).

## [0.9.1] - 2026-11-01

### Fixed
- `FairnessMonitor`: pairs now join decisions of the same batch (or of the previous one). The unpaired reference
  decisions used to accumulate when the reference group is larger, so a group's new decisions were paired with old
  reference decisions; a change shared by every group (e.g. approvals rising for everyone) then appeared as a false
  disparity. Found on a week-long demonstration stream (the bias-injection bench used stationary rates); regression
  test added. Bench rerun: false alarms 0 to 1 %, power nearly unchanged.

## [0.9.0] - 2026-11-01

### Added
- `pulsar_metrics.fairness.counterfactual_analysis`: analysis of counterfactual probes of LLM applications and agents
  (rows from `pulsar_data_collection.genai.probes`). Flip rate (Wilson), its excess over the placebo flip rate
  (Newcombe; noise floor of the instrument), and the difference and ratio (4/5 rule) of the rate of a given action,
  paired by set (BCa bootstrap, McNemar exact test, Benjamini-Hochberg over groups). Failed requests are excluded.
- `pulsar_metrics.fairness.comparative_analysis`: comparative probes asked in both orders. Position effect (binomial
  test, instrument control), tie rate, and order-balanced preference for each group.

## [0.8.0] - 2026-11-01

### Added
- `pulsar_metrics.explain`: explanations of a model and their drift.
  - `tree_attributions` (TreeSHAP, optional `explain` extra) and `importance` (share of the total attribution).
  - `AttributionDrift`: NDCG@k and top-k overlap of the feature ranking, and change of each feature's share of
    importance, against a reference window; bootstrap intervals (simultaneous over features), verdict from the
    interval, and no verdict when the ranking is not stable on the reference window (instrument check).
  - `importance_by_group`: share of importance per group minus the reference group's, stratified simultaneous
    bootstrap, small-cell suppression.
  - `explain_decision` and `describe`: plain-language sentences in French and English.
- New metric type `explainability`.
- Validated on ACSIncome (Folktables, MA 2018): no alarm under the null over 20 random splits; a model retrained on
  labels biased against women is caught by the share of `SEX` (+10.9 points [+10.5; +11.2]).

## [0.7.0] - 2026-11-01

### Added
- `pulsar_metrics.fairness.FairnessMonitor`: continuous, anytime-valid fairness monitoring of a stream of decisions.
  One sequential test by betting per monitored group against the reference group (Chugg et al., NeurIPS 2023; Online
  Newton Step bets, two-sided, with a tolerance), `alpha` split over the groups; criteria `demographic_parity` and
  `equal_opportunity`. The false alarm probability stays below `alpha` however often the monitor is looked at.
  Results are `MetricResults` with `test_type="sequential"` and the `e_value` at which the disparity was established.
- `pulsar_metrics.fairness.BettingTest`: the underlying two-sample test for observations in [0, 1].
- `benchmarks/fairness_drift.py`: bias-injection bench (false alarms under the null, power, detection delay, against a
  fixed-sample test re-run after every batch), on synthetic data and on ACSIncome (Folktables).

## [0.6.0] - 2026-11-01

### Added
- `pulsar_metrics.fairness`: group fairness of decisions. Criteria: disparate impact ratio (4/5 rule), demographic
  parity, equal opportunity, false positive rate, equalized odds, calibration and error-rate differences, each against
  a reference group (default: the largest). Prediction-based criteria by default; label-based ones on request.
  Stratified bootstrap confidence intervals, simultaneous over groups (max-t band); three-way verdict (established /
  within the zone / inconclusive) from the interval, never from the point estimate alone; suppression of groups below
  `min_group_size`; intersectional groups; in-memory join of protected attributes on a pseudonymous id.
- `MetricResults`: optional group-level fields `group`, `reference_group`, `n`, `ci_low`, `ci_high`, `method`,
  `test_type`, `e_value`, `suppressed`; new metric type `fairness`.
- Validated against Fairlearn 0.15 on ACSIncome (Folktables, MA 2018): identical values.

## [0.5.1] - 2026-11-01

### Changed
- **pandas 3 supported** (`pandas>=2.2,<4`); the lock file now uses pandas 3.1. All tests pass with pandas 3.1 and
  2.3, characterization values unchanged at 1e-9.
- **Python 3.14 and 3.15 supported and tested.** Python 3.15 requires pandas 3 (pandas 2 has no build for it).
- Continuous integration: Python 3.12, 3.13, 3.14 and 3.15 with pandas 3, plus Python 3.12 with pandas 2.
- pytest no longer references `SettingWithCopyWarning`, removed in pandas 3 (copy-on-write); a dedicated test keeps
  checking it under pandas 2.

## [0.5.0] - 2026-11-01

### Added
- `AACalibrator` (`pulsar_metrics.calibration`): thresholds calibrated on A/A draws from the reference (same system, no
  change), per metric and feature, with empirical p-values for any statistic. Contiguous windows by default to keep the
  temporal dependence of real traffic. On an AR(1) stream (phi = 0.9), false alarms of `ks_2samp` fall from 71 % with
  p < 0.05 to 2 % with the calibrated threshold. Warns when the reference holds fewer than 10 windows (the A/A draws
  then overlap and the thresholds are unreliable).
- `benjamini_hochberg` and `fdr_correct` (`pulsar_metrics.multiple_testing`): false discovery rate control across all
  the features tested in a window.
- `Analyzer.run(options=...)` accepts per-feature keys `"metric|feature"`, which take precedence over per-metric keys.

## [0.4.0] - 2026-11-01

### Added
- `DriftPersistence` (`pulsar_metrics.analyzers.persistence`): confirms a drift only when it is seen in `windows`
  consecutive analysis windows (default 2) and clears it after as many windows without drift. Reduces single-window
  false alarms in continuous monitoring, at the cost of `windows - 1` windows of delay. Motivated by a 6-minute
  end-to-end monitoring test in which an unchanged feature was flagged on one window only (#38).

### Changed
- License metadata aligned with the `LICENSE` file: the package is distributed under the **Apache License 2.0**
  (`pyproject.toml` previously declared MIT).

## [0.3.1] - 2026-11-01

### Fixed
- `psi` returned `inf` when a bin was empty in the reference sample and silently skipped bins empty in the current
  sample. It now uses the shared counts and the same additive smoothing as `kl` (`epsilon=1e-6`) (#32).
- `Analyzer.run()` failed on ISO timestamps with mixed precisions (with and without fractional seconds), as read back
  from InfluxDB: pandas 2 infers the format from the first element. Timestamps are now parsed with `format="mixed"` (#33).
- Summary statistics failed on date columns (most frequent value not accepted by `MetricResults`) and boolean columns
  (treated as numeric). Booleans now follow the categorical branch and the most frequent value is stored as text (#34).
- Summary statistics failed for every column of the current dataset absent from the reference (InfluxDB tags,
  predictions, technical columns). Such columns now get their statistics without reference value (#35).

### Documentation
- README rewritten for the current API: quick start, result structure and verdict convention, metrics reference
  (including `psi`, `mmd`, `bftest`), Analyzer columns and options, custom metrics, development.

## [0.3.0] - 2026-11-01

### Changed
- **`drift_status` has one meaning for every metric: `True` = drift or degradation detected, `False` = none, `None` =
  no verdict.** Distance metrics (`psi`, `kl`, `wasserstein`, `mmd`) and custom metrics previously returned `True` when
  the value was *below* the threshold, i.e. when there was no drift (#31).
- Performance metrics infer the direction of the threshold: minimum for accuracy, precision, recall, f1, auc, aucpr,
  r2; maximum for log_loss, brier, mse, mae, mape. `upper_bound` overrides it (#31).
- A metric evaluated without threshold returns its value with `drift_status=None` instead of failing (#31).
- Errors are explicit: metrics raise `MetricEvaluationError` instead of printing and returning nothing;
  `Analyzer.run()` records a failing metric with the new `MetricResults.error` field and evaluates the remaining ones
  instead of stopping at the first exception (#31).
- `add_drift_metrics` and `add_performance_metrics` reject unknown metric names (unknown performance metrics were
  silently ignored) (#31).

### Fixed
- `chi2` applied `chisquare` to raw samples and failed as soon as their sizes differed. It is now a test of homogeneity
  on the 2 x k contingency table (`chi2_contingency`) (#30).
- `kl` used the element-wise `kl_div` and returned an array. It now computes $D_{KL}$(current || reference) on binned
  distributions with additive smoothing (#30).
- `mape` computed the mean absolute error; it now computes the mean absolute percentage error (#30).
- `get_population_percentages(binned=True)` swapped the `ref` and `new` labels (#30).
- The most frequent value of categorical features was computed but never added to the summary statistics (#31).

## [0.2.0] - 2026-11-01

### Changed
- Python 3.12 or later (tested on 3.12 and 3.13), pandas 2, numpy 2, pydantic 2, scipy >= 1.13,
  scikit-learn >= 1.5. Numerical results are unchanged (characterization tests at 1e-9) (#27).
- Packaging: PEP 621 `pyproject.toml` with hatchling, uv and `uv.lock`; poetry removed. `black` and `DateTime` are no
  longer runtime dependencies (#27).
- Continuous integration runs ruff, black and pytest on Python 3.12 and 3.13; tests are required to pass (#28).

### Fixed
- The installed package could not be imported (`import constant` was absolute) (#25).
- `Analyzer.results_to_pandas()` always returned `None` (#25).
- `add_drift_metrics()` without `features_list` read a non-existent attribute; `features_list` is now required (#25).
- On Python 3.13 the metric enumerations were empty (`functools.partial` is no longer an `Enum` member) (#28).

### Added
- Characterization tests freezing the output of every metric (`tests/test_characterization.py`) (#26).

## [0.1.3] - 2023-02-28

Last release published on PyPI (Python 3.9).
