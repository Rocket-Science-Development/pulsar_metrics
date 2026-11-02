# Changelog

All notable changes to pulsar-metrics. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - genai-fairness

Fixes for the limits confirmed by the night lab (report `rapport_nuit.md`, section 2) on `genai` and `fairness`.

### Added
- `genai.component_test` / `embedding_drift(test="component")`: shift of the mean PCA components (Hotelling with a
  ridge, or per-component max-t that names the component), exact permutation p-value (TX-02 M6).
- `genai.novelty_test` / `embedding_drift(test="novelty")`: k-nearest-neighbour novelty score per text, threshold
  calibrated on a held-out part of the reference, permutation test of the share of novel texts; the novel texts are
  returned (TX-02 M7: 5 % of foreign texts, synthetic replica 0.8 power at 300 texts against 0.2 for the MMD).
  The two new results use metric_type "custom" until their names join the drift metric list.
- `genai.detection_power`: detection rate by share of changed texts (or images) and window size, with the A/A rate:
  the minimal detectable change per window size (TX-02 M6/M7, IM-02 m5).
- `mmd_test(bandwidth=...)`: fixed kernel unit instead of the median distance.
- `genai.model_equality_power`, `genai.RECOMMENDED_CANARY` (48 prompts x 10 answers) and
  `model_equality(min_answers=...)`, which reports a non-significant result on a small canary as inconclusive; the
  `method` of a non-significant result below 480 answers says that a quantization change is usually missed (LM-01 M23).
- `ComplianceMonitor.alerts(trace, min_severity="critical")`: per-run alert of the violated rules by severity, with the
  rules version; `ComplianceMonitor.run_record(trace)`: `violated_rules`, `critical_violations`, `max_severity`,
  `rules_version` for the agent_run record; `genai.SEVERITIES` (info, minor, major, critical; unknown severities warn
  and rank as major) (LI-03 C3).
- `ComplianceMonitor.evaluate(reference=..., alpha=0.05, absolute_from="critical")`: rules judged against a reference
  period (one-sided Fisher exact test, Benjamini-Hochberg over the rules); critical rules keep the absolute verdict
  (LS-01 M26).
- `FairnessAnalyzer(max_intersection_depth=2, max_cells=None, declared_groups=None)` and `FairnessAnalyzer.coverage()`
  (share of rows without verdict per feature and criterion, `no_verdict_share`), with a warning above
  `FairnessPolicy.max_no_verdict_share` (5 %) of rows in suppressed cells (FA-01 M8, m6, m7).
- `FairnessPolicy(interval="auto", small_group_size=200)`: Wilson score intervals (Newcombe for differences, MOVER-R
  for the ratio) for small groups on the rate criteria (FA-04 m8); `FairnessPolicy(equivalence_confidence=...)` and
  `counterfactual_analysis(equivalence_confidence=...)`: TOST for the "within the zone" verdicts (FA-04 m9, m10).
- Inconclusive verdicts of `FairnessAnalyzer` and `counterfactual_analysis` give in `method` the rows or probe sets
  needed for an 80 % chance of a verdict (FA-01 m7, FA-04 m9, m10, LB-01 m17).
- `pulsar_metrics.fairness.planning`: `required_group_size`, `paired_sets_needed`, `flip_sets_needed`,
  `sequential_decisions_needed`, and the score intervals `wilson`, `newcombe_difference`, `mover_ratio`,
  `rate_interval`; `fairness.equivalence_verdict`.
- `FairnessMonitor(baseline=...)`, `FairnessMonitor.fit_baseline(frame)`: change mode, H0 |gap - baseline| <= tolerance
  (FA-01 M9); `FairnessMonitor.acknowledge(group, rebaseline=None)` re-arms a test after an alarm (`alarms` keeps
  the history); `BettingTest(center=...)`.

### Changed
- **Default changed** `FairnessAnalyzer(max_intersection_depth=2)`: intersections stop at pairs of attributes. All
  combinations (the former behaviour) exploded with 5 attributes (65 to 75 % of the cells suppressed, 30 to 80 times
  the run time); pass `max_intersection_depth=None` to keep them. Combinations whose cells are all suppressed are
  skipped with a warning instead of raising.
- **Default changed** `FairnessPolicy(interval="auto")`: groups under 200 rows get score intervals on the rate criteria;
  the percentile bootstrap was anti-conservative for them (3 to 5 % one-sided misses instead of 2.5 %).
  `interval="bootstrap"` restores the former intervals.
- **Default changed** `FairnessMonitor(reference_pooling=True)`: each group decision is paired with the mean of a
  disjoint chunk of the batch's reference decisions; the 1:1 pairs dropped about 95 % of the reference decisions of a
  small group (FA-04 m11: group of 5 %, drop of 0.1, 20,000 decisions, detection 16 % -> 52 %, false alarms 0 to 0.5 %).
  `reference_pooling=False` restores the 1:1 pairs. The README example now establishes the disparity after 3,138
  pairs (e-value 22) instead of 3,184 (21).
- The fairness bootstrap of 0/1 rates draws exact binomial replicates instead of resampling rows in a Python loop
  (same distribution; 5 attributes x 50,000 rows, all combinations: 188 s -> 5 s). Bootstrap intervals differ in their
  last digits from 0.10.
- `mmd_test` evaluates the permutations with one matrix product per block and fits the PCA by eigendecomposition
  (TX-02 m4: 1000 x 1000 texts, 500 permutations, 11 to 18 s -> 0.7 s); same p-values for the same seed.

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
