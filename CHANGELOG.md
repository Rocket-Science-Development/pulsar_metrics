# Changelog

All notable changes to pulsar-metrics. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
