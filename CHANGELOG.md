# Changelog

All notable changes to pulsar-metrics. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.14.0] - 2026-11-05

### Added

- **Data quality against a reference profile** (`pulsar_metrics.quality`). `DataQualityProfile.fit` learns, on a
  reference window, the kind, dtype, missing share, range and robust quantiles, categorical domain and cardinality of
  each column, plus the duplicate-row share and optional row rules (`"dew <= temp"`); the profile serializes to JSON or
  YAML (`pulsar.dq-profile/1`) to be versioned with the model. Six checks compare a window to the profile, each a
  `MetricResults` with `metric_type="data_quality"`: `schema` (missing, new or retyped column), `null_share`,
  `out_of_range_share`, `unseen_category_share`, `duplicate_share`, `rule_violation_share`. Thresholds per check and per
  column in a `DataQualityPolicy` (dict or YAML), absolute (`{max: x}`) or relative to the reference (`{margin: x}`).
  A check that cannot run has no value nor verdict and a status code; a window without rows makes every check, the
  schema included, `empty_window`. Infinite values are in range when the reference held the same infinity, column
  names are read as text, `range_quantiles` levels are checked against the profile's when the policy is loaded. `DataQualityAnalyzer` runs them like the other
  analyzers; `to_evidence` exports them in the `data_quality` family with seven new reliability codes
  (`empty_window`, `missing_in_window`, `not_in_reference`, `type_mismatch`, `no_reference_values`,
  `no_reference_domain`, `no_values_in_window`). On the open hourly weather of Montréal-Trudeau (ECCC), no false
  positive over 36 clean months with a per-column policy, and each provoked break caught by its check.
- **Consistency of LLM answers** (`pulsar_metrics.genai.consistency`, `consistency_drift`): mean pairwise agreement
  and share of questions in full agreement on a window, with bootstrap intervals over the questions, and a one-sided
  permutation test of a drop against a reference (paired on shared questions); a missing `full_agreement` is read
  from the agreement, a non-boolean one refused. Exported in the `genai` family, with
  the reliability codes `few_repeats` and `few_questions`. On 30 questions with local models, a temperature rise is
  detected (agreement −0.19 and −0.30) and an A/A split is not (4.4 % false alarms over 500 splits). Answers are
  collected by `pulsar_data_collection.genai.ConsistencyProbe`.
- **`rate_increase` drift test**: one-sided exact Fisher test on a 0/1 indicator per run (abstentions, answers given
  without support). Exact on small windows, accepts a reference with no positive without flagging it
  `constant_reference`, and exports both rates in `details` of the evidence records.

## [0.13.0] - 2026-11-04

### Added

- `pulsar_metrics.evidence`: export of results in the versioned format `pulsar.evidence/1`, for a service that turns
  measurements into evidence. Typed pydantic records with a deterministic `evidence_id` (deduplication), system,
  model, metric with threshold, direction and status, window and reference, a reliability block with machine-readable
  codes (blocking `reasons`, advisory `notes`), A/A calibration, persistence and data quality, provenance, and
  `details` for aggregates only (64 KiB cap). Converters from an Analyzer, the fairness analyzers, the GenAI and
  compliance results and any list of `MetricResults` (`to_evidence`); JSON round trip; JSON Schema shipped in
  `pulsar_metrics/schemas/evidence-1.json` and checked by the tests. Compatibility: additive changes stay in `/1`,
  breaking changes move to `/2`. `producer_version` is read from the source tree when the package runs from a
  checkout (editable installs keep stale metadata).
- `mmd_test` and `embedding_drift(test="mmd")` accept `groups=(reference_groups, current_groups)`: permutations and
  subsampling move whole groups of rows (for example the passages retrieved by one agent run) instead of rows taken
  as independent. On the LR-03 replay (windows of 100 runs), the A/A false-alarm rate of MMD on retrieved passages
  goes from 41–43 % to 5.0 % [4.3; 5.8]. Without `groups`, behaviour and p-values are unchanged. The test assumes the
  groups are exchangeable between the windows, sizes included; the subsampling keeps at least two whole groups per
  window (even above `max_samples`), and a window with a single group is refused with a message that says so.
- Review of the evidence export before release: `window.id` (optional, hashed) for results without timestamps; a
  record with no window start, end nor id hashes `produced_at` too, with a `UserWarning`, so two windows no longer
  share an id; the rules version of compliance records is hashed (`details.rules_version`); `from_dict` / `from_json`
  recompute `evidence_id` and refuse a mismatch; NaN and infinities are refused by the models, `loads()` and
  `dumps()`; a zone bound may be `null` (unbounded: the default zone of `error_rate_difference` no longer breaks the
  export); NaT window bounds are read as missing; the direction follows the exported threshold (a NaN threshold gives
  `none`); `position_effect` is exported as its two-sided binomial test (`p_value` basis, threshold alpha, p-value,
  `details.null_value` 0.5); `unusable_rate` is decided on its `value`; `no_verdict_share` above the policy maximum
  is an `alert`; the calibration block reports `nominal` when the threshold used is not the calibrated one; new reason
  codes `non_finite_value` and `unstable_reference`; fairness criteria and performance metrics without labels are
  `not_evaluable` with `no_ground_truth`.

### Changed

- The Analyzer keeps the window and reference sizes and bounds of its last run (`_window_info`), and
  `DriftPersistence` keeps whether the last window changed the streak (`changed`), for the evidence export. The
  columns of `results_to_pandas` are unchanged. A run refused for its windows (no row of the model) no longer
  replaces the period of the previous results.
- Results that feed the export: `position_effect` carries the `p_value` of its binomial test and alpha in `method`;
  the NDCG and Jaccard results of `AttributionDrift` on an unstable reference have `status="unstable_reference"` and
  `reliable=False`; a fairness criterion skipped for lack of labels, and a performance metric without a label column,
  carry `n_labelled=0`.

## [0.12.0] - 2026-11-04

Changes recommended by the P2 lab of 3 November 2026 (TC-02, TC-04, TS-04, FA-03, LA-02, LB-02).

### Added

**Calibration and drift (TC-02, TC-04, TS-04)**

- `AACalibrator(compare_to="reference" | "window")`. A null whose A/A draws sit at the bound of the statistic in more
  than 5 % of draws (p-value underflowed to 0, psi/kl at the epsilon ceiling) gets the status `saturated`, a warning
  advising wasserstein or a reference per regime, and `reliable=False` on its results. `NullDistribution.share_at_bound`.
- `NullDistribution.between_within_ratio`; `fit()` warns when it is more than 10 times what
  the within-window lag-1 autocorrelation implies (the reference is dominated by seasons or regimes).
- `period_reference(reference, current, weeks=4, time_column=...)`: the reference rows in the same weeks of the year,
  to calibrate and compare seasonal data.
- Optional grouping of sparse levels for chi2, psi and kl (`min_expected`, `max_levels`, also in
  `Analyzer.add_drift_metrics`), chosen on the reference and applied alike to the window and the A/A draws.
- `MetricResults`: `reliability_reason`, `low_expected_share` (share of cells with an expected count below 5); drift
  results fill `n` (rows of the window, missing values included). A categorical feature above 20 % sparse cells without grouping warns.

**Fairness on aggregate data (FA-03)**

- `FairnessAnalyzer(weights=, unit=)`: row weights used by every group rate and interval. Many-to-one composition
  tables: `run(attributes_data=..., on=..., share=...)` or `count=...`, converted to weights; results built from
  aggregates carry `status="ecological"`.
- `cluster=`: the bootstrap resamples units, jointly across groups; `n` and `min_group_size` count units. A warning
  flags mostly duplicated rows.
- `bins=` discretises a numeric protected attribute by quantiles or edges (`bin_edges`); a numeric attribute with more
  than `numeric_levels=20` distinct values warns.

**Comparative probes (LB-02)**

- `comparative_analysis` publishes `unusable_rate` (errors, refusals `choice="refusal"`, unparsed answers) and a
  `comparative_instrument` result whose status is `instrument_unusable` when answers are mostly unusable or order-driven
  (`max_unusable_rate`, `max_order_driven_share`).

**Agent compliance (LA-02)**

- Rules read rich traces (`{name, kind, args, status}` elements and conversation events), declared by
  `ComplianceMonitor.accepts_rich_trace = True`. Atom predicates `tool[arg=value]`, `tool[arg!=value]`, `tool[arg]`,
  `tool[status=error]`, events `@name`, wildcards `*[...]` and `@*`; lists of atoms ("any of") in `events`.
- Templates `alternate_precedence` and `alternate_response`; argument binding `bind: [order_id]` (the rule holds for
  every value), usable from the YAML file.
- "Not evaluable" outcome: `Rule.verdict()` (True, False or None), `RuleNotEvaluable`, `NotEvaluableWarning`,
  `NOT_EVALUABLE`, `ComplianceMonitor.not_evaluable()`, `not_evaluable_rules` in `run_record`, `status="not_evaluable"`
  in `evaluate`. `trace_from` returns a `NameTrace` (a list). On a synthetic replica of LA-02, the confirmation rule goes
  from 36 structural false alarms to the 23 real violations.

### Changed

- `AACalibrator` compares by default each A/A window with the rest of the reference (`compare_to="reference"`), as
  `Analyzer.run` does. Window-against-window thresholds were too lenient for that comparison (AR(1), phi = 0.9:
  0.2 % wasserstein false alarms instead of 5 %, hence a loss of power). `compare_to="window"` keeps the old draws, and
  calibrations saved before load back that way; the cache format is `pulsar-aa-null-v2`. Calibration is slower (about
  7x on a reference of 40 windows).
- `auc` and `aucpr` need 30 labelled rows and both classes by default (`min_labelled=1` restores the old behaviour);
  other performance metrics are `reliable=False` below 30 labelled rows. A drift window more than 2x away from the
  calibrated size is `reliable=False`, and `empirical_pvalues()` warns.
- `comparative_analysis` results start with `unusable_rate` and include `comparative_instrument`; when every answer is
  unusable it returns them instead of raising.
- `FairnessAnalyzer` raises an explicit `CustomExceptionPulsarMetric` instead of a pandas `MergeError` on a many-to-one
  join without `share=`/`count=` or on duplicate prediction ids; `AllCellsSuppressed` on a numeric attribute suggests
  `bins=`.

### Fixed

- `position_effect` counted a `choice` other than first/second/tie as a non-first decision; such answers are excluded,
  also from the `tie_rate` denominator.
- `allowed_tools` raised on rich trace elements; it now checks tool names and ignores events.

## [0.11.0] - 2026-11-02

Fixes for the limits confirmed by the night lab of 2 November 2026: C1, C2, C3 (metrics side), M1 to M9, M23
(model equality side), M26, m1 to m11, m16, m17 and m21. Embedding mean-shift and novelty results are drift metrics.

### Added

**Calibration, analyzer and drift/performance metrics (TS-01, TS-02, TC-01, TC-05, SC-06, LS-01)**

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

**GenAI, compliance and fairness (TX-02, IM-02, LM-01, FA-01, FA-04, LB-01, LI-03, LS-01)**

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
- `FairnessPolicy(interval="auto", small_group_size=200)`: Wilson score intervals (Newcombe for differences, MOVER
  on the log scale for the ratio) for small groups on the rate criteria (FA-04 m8); `FairnessPolicy(equivalence_confidence=...)` and
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

**Calibration, analyzer and drift/performance metrics (TS-01, TS-02, TC-01, TC-05, SC-06, LS-01)**

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

**GenAI, compliance and fairness (TX-02, IM-02, LM-01, FA-01, FA-04, LB-01, LI-03, LS-01)**

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
  (TX-02 m4: 1000 x 1000 texts, 500 permutations, 11 to 18 s -> 0.7 s); same p-values for the same seed. The PCA
  keeps only directions with positive variance, so at most min(n_components, rank) components.
- Missing values of a protected attribute (NaN, None) now form a group of their own, `<missing>`, in
  `FairnessAnalyzer`; they used to crash it on pandas 3 (`astype(str)` keeps NaN there).

### Fixed

**GenAI, compliance and fairness (TX-02, IM-02, LM-01, FA-01, FA-04, LB-01, LI-03, LS-01)**

- PCA of the embedding tests with fewer texts than components and more dimensions than texts: the numerically null
  directions of the Gram matrix were divided by ~0 (projections up to 1e137, wrong p-values in about half of the
  draws at 12 + 12 texts in 384 dimensions).
- Ratio score interval: MOVER on the log scale instead of the direct MOVER-R formula, which returned a lower bound of
  0 when the upper Wilson bound of the reference rate exceeded twice that rate (observed ratio 10: [0, 59], now
  [1.7, 61]).
- `FairnessAnalyzer`: only combinations whose cells are all suppressed (`AllCellsSuppressed`) are skipped; other
  errors propagate. The `method` of exploratory (non-declared) groups names their interval instead of a dict.

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
