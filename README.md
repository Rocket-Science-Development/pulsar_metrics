# pulsar-metrics

[![Continuous Integration](https://github.com/Rocket-Science-Development/pulsar_metrics/actions/workflows/continuous-integration.yaml/badge.svg)](https://github.com/Rocket-Science-Development/pulsar_metrics/actions/workflows/continuous-integration.yaml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue)
![pandas](https://img.shields.io/badge/pandas-2.2%2B%20%7C%203.x-blue)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**pulsar-metrics** is an open-source Python library for evaluating and monitoring **data drift**, **concept drift** and
**model performance**. Every metric returns the same result structure with a value, an explicit drift verdict and,
when something goes wrong, an error message instead of a silent failure. User-defined metrics are supported.

It applies to classic models as well as to **LLM applications and agents**: their requests, answers and trajectories
are turned into features (steps, tools called, errors, tokens, latency...) and compared between windows with the same
tests, under thresholds calibrated for non-deterministic traffic.

![Pulsar.ml architecture](https://raw.githubusercontent.com/Rocket-Science-Development/pulsar_demo/main/images/pulsarml-architecture.png)

It pairs with [pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection), which
collects features and predictions from a model serving code into InfluxDB.

- [Installation](#installation)
- [Quick start](#quick-start)
- [Results and drift verdicts](#results-and-drift-verdicts)
- [Available metrics](#available-metrics)
- [Computing metrics one by one](#computing-metrics-one-by-one)
- [The Analyzer](#the-analyzer)
- [Calibrated thresholds and multiple testing](#calibrated-thresholds-and-multiple-testing)
- [Confirming drift over successive windows](#confirming-drift-over-successive-windows)
- [Data quality against a reference profile](#data-quality-against-a-reference-profile)
- [Monitoring LLM applications and agents](#monitoring-llm-applications-and-agents)
- [Group fairness](#group-fairness)
  - [Continuous fairness monitoring](#continuous-fairness-monitoring-anytime-valid)
  - [Counterfactual probes](#counterfactual-probes-of-llm-applications-and-agents)
  - [Text drift, model equality, compliance](#text-drift-model-equality-and-agent-compliance)
- [Explanations and their drift](#explanations-and-their-drift)
- [Custom metrics](#custom-metrics)
- [Development](#development)
- [Changelog](CHANGELOG.md)

## Installation

Requires **Python 3.12 or later** (tested on 3.12, 3.13, 3.14 and 3.15).

```bash
pip install "pulsar-metrics @ git+https://github.com/Rocket-Science-Development/pulsar_metrics.git"
```

Pin a commit or a tag for reproducible installs (`...pulsar_metrics.git@<sha>`). The PyPI release (0.1.3) predates the
current API and only supports Python 3.9.

Main dependencies: pandas 2.2+ or 3.x, numpy, scipy, scikit-learn, pydantic 2. Python 3.15 requires pandas 3 (pandas 2
is not built for it).

## Quick start

Compare a current window of predictions with a reference window (typically a hold-out set scored at training time):

```python
import numpy as np
import pandas as pd

from pulsar_metrics.analyzers.base import Analyzer

rng = np.random.default_rng(0)


def window(n, shift, start):
    x = rng.normal(shift, 1, n)
    return pd.DataFrame(
        {
            "model_id": "credit_model",
            "model_version": "1.0",
            "pred_timestamp": pd.date_range(start, periods=n, freq="min"),
            "income": x,
            "channel": rng.choice(["web", "branch"], size=n, p=[0.7, 0.3] if shift == 0 else [0.4, 0.6]),
            "y_true": (x + rng.normal(0, 1, n) > 0).astype(int),
            "y_pred": (x > 0).astype(int),
        }
    )


reference = window(1000, 0.0, "2026-01-01")
current = window(500, 0.8, "2026-02-01")

analyzer = Analyzer(name="weekly check", model_id="credit_model", model_version="1.0")
analyzer.add_drift_metrics(metrics_list=["ks_2samp", "wasserstein", "psi"], features_list=["income"])
analyzer.add_drift_metrics(metrics_list=["chi2"], features_list=["channel"])
analyzer.add_performance_metrics(metrics_list=["accuracy", "f1"])

analyzer.run(
    current=current,
    reference=reference,
    options={"wasserstein": {"threshold": 0.2}, "psi": {"threshold": 0.2}, "accuracy": {"threshold": 0.7}},
)

results = analyzer.results_to_pandas()
metrics = results[results.metric_type != "statistics"].astype({"metric_value": float})
print(metrics[["metric_name", "feature_name", "metric_value", "drift_status"]].to_string(index=False, float_format="{:.3g}".format))
```

```
metric_name feature_name  metric_value drift_status
   ks_2samp       income      3.63e-40         True    <- p-value below alpha (0.05)
wasserstein       income         0.905         True    <- above the 0.2 threshold
        psi       income         0.822         True
       chi2      channel      2.67e-30         True
   accuracy   prediction         0.834        False    <- above the 0.7 minimum
         f1   prediction         0.889         None    <- no threshold given: value only
```

## Results and drift verdicts

Every metric produces a `MetricResults` (pydantic model):

| Field | Meaning |
|---|---|
| `metric_type` | `drift`, `performance`, `statistics`, `data_quality` (see [Data quality](#data-quality-against-a-reference-profile)) or `custom` |
| `metric_name` | e.g. `ks_2samp`, `psi`, `accuracy` |
| `feature_name` | Feature analyzed; `prediction` for performance metrics |
| `metric_value` | Value of the metric (p-value for statistical tests) |
| `threshold` | Threshold, `alpha` or `[min, max]` interval used for the verdict |
| `drift_status` | **`True` = drift or degradation detected, `False` = none, `None` = no verdict** |
| `conf_int` | Bootstrap confidence interval (performance metrics with `bootstrap=True`) |
| `error` | Message when the metric could not be computed (`metric_value` and `drift_status` are then `None`) |
| `status` | Data-quality status of a drift result: `constant_reference` when the reference of the feature holds a single value, `saturated` when the calibrated null sits at the bound of the statistic (see [Known limits of the calibration](#known-limits-of-the-calibration)) |
| `missing_share_current`, `missing_share_reference` | Drift metrics: share of missing values (NaN or declared sentinels) set aside |
| `n_labelled`, `label_coverage` | Performance metrics: rows evaluated (ground truth and prediction present) and their share of the window |
| `reliable` | `False` when the threshold comes from a calibration flagged unreliable (`AACalibrator.options()`), when the window is more than 2x smaller or larger than the calibrated `window_size`, or when a performance metric rests on fewer than 30 labelled rows: provisional verdict |
| `reliability_reason` | Why `reliable` is `False` |
| `n` | Drift metrics: rows of the current window behind the value (after missing values are set aside) |
| `low_expected_share` | `chi2`, `psi`, `kl`: share of the cells of the 2 x k table with an expected count below 5 |

The verdict follows **one convention for every metric**:

- **Statistical tests** (`DriftTestMetric`): drift when the p-value is below `alpha` (default 0.05).
- **Distance metrics** (`DriftMetric`) and custom metrics: drift when the value **exceeds** the threshold
  (`upper_bound=True`, default), falls below it (`upper_bound=False`), or lies **outside** a `[min, max]` interval.
- **Performance metrics**: the threshold is a **minimum** for accuracy, precision, recall, f1, auc, aucpr and r2, and a
  **maximum** for error metrics (log_loss, brier, mse, mae, mape). Pass `upper_bound` to override.
- **No threshold**: the value is computed and `drift_status` is `None`.
- **NaN statistic**: no verdict. `drift_status` is `None` and `error` says so (comparing NaN with a threshold used to
  give `True` for distances and `False` for tests).

Errors are explicit: a metric that cannot be computed (missing column, incompatible types, failing function) raises
`pulsar_metrics.exceptions.MetricEvaluationError` with the original cause. `Analyzer.run()` does **not** stop on such a
failure: it logs a warning (logger `pulsar_metrics.analyzers.base`), records a result with `error` set, and evaluates
the remaining metrics.

## Available metrics

### Data drift: distances (`DriftMetric`, verdict on `threshold`)

| Name | Metric | Notes |
|---|---|---|
| `psi` | [Population Stability Index](https://www.listendata.com/2015/05/population-stability-index.html) | $\sum (p_{cur} - p_{ref}) \ln(p_{cur}/p_{ref})$ on shared bins (numeric, Sturges rule) or categories. Common reading: < 0.1 stable, 0.1–0.2 moderate, > 0.2 significant |
| `kl` | [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) $D_{KL}(P_{cur} \parallel P_{ref})$ | Same binning as `psi`; zero for identical distributions |
| `wasserstein` | [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric) $W_1$ | In the units of the feature: standardize features, or set thresholds per feature |
| `mmd` | [Maximum Mean Discrepancy](https://jmlr.org/papers/v13/gretton12a.html) | Linear kernel by default (`kernel=` accepts scikit-learn pairwise kernels) |

`psi` and `kl` add a small smoothing (`epsilon=1e-6`, then renormalisation) to every proportion, so that a category or
bin absent from one sample gives a large but finite value instead of `inf`. `psi`, `kl` and `chi2` bin numeric
features as float whatever their dtype, so that an integer column and the same column as float give the same value.
The smoothing also caps them: two samples without any shared bin give `psi` = 2 ln(1/epsilon) = 27.6, however large
the shift.

**Many sparse levels.** With hundreds of levels (airports, products), most cells of the table are nearly empty:
`chi2` spends its degrees of freedom on them and loses power, and `psi` / `kl` add noise. `min_expected=5` merges the
levels whose expected count in the current window (current size x reference share) is below 5 into one `__other__`
level; `max_levels=k` keeps the k - 1 most frequent reference levels. The levels are chosen on the reference, so the
same grouping applies to the window and to every A/A draw of a calibration. Both are opt-in, at construction or in
`add_drift_metrics(..., min_expected=5)`. Every `chi2` / `psi` / `kl` result reports `low_expected_share`, and a
categorical feature above 20 % without grouping raises a UserWarning. In the lab (TC-02, 293 airports, 5 % of the rows
moved onto the 5 largest), grouping raised the power of `chi2` from 0.05 to 0.275 at 2,000 rows and from 0.255 to
0.695 at 5,000; at 20,000 rows both are near 1. Grouping is not the default: with a calibrated null the raw test keeps
a valid false alarm rate, it only loses power, and grouping changes every categorical p-value.

### Data drift: statistical tests (`DriftTestMetric`, verdict on `alpha`)

| Name | Test | Detects |
|---|---|---|
| `ttest` | [Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test) | Shift of the mean |
| `manwu` | [Mann-Whitney U](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test) | Shift of location (rank based) |
| `levene` | [Levene](https://en.wikipedia.org/wiki/Levene%27s_test) (centered on the mean) | Change of variance |
| `bftest` | [Brown-Forsythe](https://en.wikipedia.org/wiki/Brown%E2%80%93Forsythe_test) (Levene centered on the median) | Change of variance, robust to outliers |
| `ks_2samp` | [Kolmogorov-Smirnov](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) | Any change of distribution |
| `CvM` | [Cramér-von Mises](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion) | Any change of distribution |
| `chi2` | [Chi-squared test of homogeneity](https://en.wikipedia.org/wiki/Chi-squared_test) | Change of the frequencies of a **categorical** feature (2 x k contingency table, `scipy.stats.chi2_contingency`); samples may have different sizes; numeric features are binned |
| `missing_rate` | [Fisher exact test](https://en.wikipedia.org/wiki/Fisher%27s_exact_test) on the 2 x 2 table missing / present | Change of the **share of missing values** (NaN and declared sentinels), e.g. a sensor that fails more often |
| `rate_increase` | One-sided [Fisher exact test](https://en.wikipedia.org/wiki/Fisher%27s_exact_test) on a 0 / 1 indicator | **Rise** of the share of positives of a per-run indicator (e.g. `det_abstention`, `det_unsupported_answer` of a RAG agent); exact on small windows and with a zero base rate |

The statistical tests other than `chi2` apply to numeric features. For a categorical feature use `chi2`, `psi` or `kl`.

### Missing values and sentinels

Drift metrics and tests take two parameters, at construction or through `Analyzer.add_drift_metrics`:

- `missing_values`: value(s) meaning "no value", such as the `-200` of a sensor, turned into NaN before any
  computation. Without it, the sentinel is analysed as a measurement: in the lab, sentinels (15 to 31 % of a week's
  values) raised 7 drift alerts over 5 weeks that disappeared once the data were cleaned.
- `nan_policy`: `"omit"` (default) sets missing values aside and records their share in `missing_share_current` and
  `missing_share_reference`; `"propagate"` passes them to the function; `"raise"` refuses them. Before, a single NaN
  made the statistic NaN and the verdict arbitrary (1080 of 1080 lab tests without a usable verdict), and `psi` and
  `kl` raised.

Setting missing values aside hides a change of their rate: add `missing_rate` to watch it.

A feature **constant in the reference** (imputed, or not yet in use) gets `status="constant_reference"`: any variation
then counts as a drift. The change is real, but read it as "the feature started varying" rather than as a shift.

### Performance (`PerformanceMetric`)

| Name | Metric | Threshold | Predictions expected |
|---|---|---|---|
| `accuracy`, `precision`, `recall`, `f1` | Classification | minimum | labels (`y_pred`) |
| `auc`, `aucpr` | ROC AUC, average precision | minimum | scores or probabilities |
| `log_loss`, `brier` | Probabilistic losses | maximum | probabilities |
| `mse`, `mae`, `mape` | Regression errors (`mape` = mean absolute percentage error) | maximum | values |
| `r2` | Coefficient of determination | minimum | values |

Ground truth often arrives late or only for part of the traffic. Performance metrics are computed on the rows that
have both ground truth and prediction, and report `n_labelled` and `label_coverage`. `min_label_coverage` (default 0.0)
and `min_labelled`, at construction or in `evaluate()`, set the minimum below which the result carries an `error`
instead of a value. `min_labelled` defaults to **30 for `auc` and `aucpr`**, which also need both classes among the
labelled rows, and to 1 for the other metrics, whose results on fewer than 30 labelled rows get `reliable=False` with a
`reliability_reason`. In the lab (TC-04), a monthly window of 10 rows with one positive got an AUC of 0.778 without any
warning; pass `min_labelled=1` to keep the former behaviour. `missing_values` declares a sentinel of `y_true` / `y_pred`. Before, a single missing label
made the whole window fail (in the lab, 45 of 45 weeks with 18 % of labels missing). A result computed on part of the
window says so in `method`, and with the default minimum of 0 a UserWarning is emitted below 50 % coverage: set
`min_label_coverage` to refuse such windows.

### Summary statistics (`FeatureSummary`)

For each feature: mean, median, std, skewness, kurtosis, P25, P95 and count for numeric features; most frequent value
and count for categorical, boolean and date features. The reference value is stored in `threshold` for comparison
(`None` when the feature is absent from the reference).
`Analyzer.run()` computes them for every column of the current dataset (`metric_type="statistics"`).

## Computing metrics one by one

```python
from pulsar_metrics.metrics.drift import DriftMetric, DriftTestMetric
from pulsar_metrics.metrics.performance import PerformanceMetric

# Statistical test: verdict on alpha
ks = DriftTestMetric(metric_name="ks_2samp", feature_name="income")
print(ks.evaluate(current=current, reference=reference, alpha=0.01))

# Distance: verdict on threshold (or a [min, max] interval)
psi = DriftMetric(metric_name="psi", feature_name="income")
print(psi.evaluate(current=current, reference=reference, threshold=0.2).drift_status)

# Performance with a bootstrap confidence interval
acc = PerformanceMetric(metric_name="accuracy", y_name="y_true", pred_name="y_pred")
result = acc.evaluate(current=current, threshold=0.7, bootstrap=True, n_bootstrap=200, alpha=0.05, seed=123)
print(result.metric_value, result.conf_int, result.drift_status)
```

Extra keyword arguments are passed to the underlying function, e.g. `kernel="rbf"` for `mmd` or `epsilon=1e-4` for
`psi` and `kl`.

## The Analyzer

`Analyzer` runs a set of metrics on a current dataset against a reference dataset in one call.

**Expected columns** in both datasets:

| Column | Use |
|---|---|
| `model_id`, `model_version` | Rows are filtered on the analyzer's `model_id` and `model_version`, compared as text (a float column of integral values also without its ".0"): a reference read back from a CSV file (`"1.0"` became the float `1.0`) is accepted, while `"1.10"` and `"1.1"` stay distinct. When a filter leaves no row, the error says how many rows each filter kept |
| `pred_timestamp` | Prediction time (dates or ISO strings, mixed precisions accepted); gives `period_start` / `period_end` |
| features | Columns named in `features_list` |
| `y_true`, `y_pred` (or names given with `y_name` / `pred_name`) | Performance metrics, evaluated on the rows with ground truth (`n_labelled`, `label_coverage`). A missing `y_true` column makes the metric fail with an explicit error |

**Methods**

| Method | Description |
|---|---|
| `add_drift_metrics(metrics_list, features_list, missing_values=None, nan_policy="omit")` | Adds every metric for every feature; unknown names raise an error. `missing_values` is one value or list for every feature, or a dict `{feature: value or list}`; declared sentinels also become NaN in the summary statistics |
| `add_performance_metrics(metrics_list, **kwargs)` | Adds performance metrics; `y_name` / `pred_name` select the columns, `missing_values` declares their sentinel, `min_label_coverage` / `min_labelled` the minimum ground truth |
| `run(current, reference, options={})` | Computes summary statistics and all metrics. `options` maps a **metric name** to its keyword arguments (`threshold`, `alpha`, `upper_bound`, `bootstrap`...); they apply to every feature for that metric |
| `get_result()` | List of `MetricResults` |
| `results_to_pandas()` | DataFrame of results plus model metadata (`model_id`, `model_version`, `period_start`, `period_end`, `eval_timestamp`) |
| `results_to_json()` | List of JSON strings |

Because `options` are set per metric name, distance thresholds are shared by all features: standardize numeric
features against the reference (`(x - mean_ref) / std_ref`) or run one analyzer per group of features.

## Calibrated thresholds and multiple testing

A nominal rule such as "drift if p < 0.05" assumes independent samples. Production traffic, and LLM or agent traffic in
particular (non-deterministic outputs, correlated requests), breaks that assumption, and the nominal rule then raises
alerts without any change. On an autocorrelated stream (AR(1), phi = 0.9, windows of 200), `ks_2samp` with p < 0.05
flagged **71 %** of A/A windows taken from the same process; with the calibrated threshold, **2 %**.

`AACalibrator` draws `n_draws` windows from the reference ("A/A": same system, no change), compares each with the rest
of the reference, as `Analyzer.run` compares a production window with the whole reference, and builds the distribution
of each statistic without drift. It gives per-feature thresholds for the Analyzer and an empirical p-value
for any statistic, on which `fdr_correct` applies the Benjamini-Hochberg correction across all features of a window.

```python
from pulsar_metrics.calibration import AACalibrator
from pulsar_metrics.multiple_testing import fdr_correct

monitor = Analyzer(name="calibrated", model_id="credit_model", model_version="1.0")
monitor.add_drift_metrics(metrics_list=["ks_2samp", "wasserstein"], features_list=["income"])

# reference sorted by time, holding at least 10 production windows (30 for stable thresholds); window_size = their size
long_reference = window(6000, 0.0, "2025-12-01")
calibrator = AACalibrator(window_size=300, n_draws=200, quantile=0.95).fit(long_reference, monitor._metrics_list)
print(calibrator.options())  # {"ks_2samp|income": {"alpha": ...}, "wasserstein|income": {"threshold": ...}}

monitor.run(current=window(300, 0.8, "2026-03-01"), reference=long_reference, options=calibrator.options())
drift = monitor.results_to_pandas().query("metric_type == 'drift'").copy()
drift["p_empirical"] = calibrator.empirical_pvalues(drift)
print(fdr_correct(drift, "p_empirical", alpha=0.05)[["metric_name", "feature_name", "p_adjusted", "drift_fdr"]])
```

- `window_size` must match the production windows: the null distribution depends strongly on the sample size. A
  window more than 2x smaller or larger gets `reliable=False` with a `reliability_reason` (`options()` carries
  `window_size`), and `empirical_pvalues()` warns: in the lab (TC-04), monthly windows of 10 to 67 rows got BH verdicts
  with thresholds calibrated on 1,000 rows.
- `compare_to="reference"` (default) compares each A/A window with the rest of the reference, which is what
  `Analyzer.run` does. `compare_to="window"` compares two disjoint windows, for a monitoring that gives the Analyzer a
  reference of one window (e.g. the previous week); it was the only mode before, and its thresholds are too lenient
  for a comparison against a long reference: on an AR(1) stream (phi = 0.9, windows of 200, 8,000 rows) they flagged
  under 2 % of fresh windows compared with the whole reference, against about 5 % with the default. Calibrations saved
  before are read back as `compare_to="window"`.
- The reference must hold **at least 10 windows**: `fit()` refuses a shorter one unless `allow_short=True` (with 3
  windows of 300 in a 1,000-row reference, the calibrated `alpha` of `ks_2samp` came out at 0.34 on independent data).
  Contiguous draws need 4 windows even then: with about 2, every draw compares nearly the same two windows (in the lab,
  1 distinct value over 30 draws and a calibrated `alpha` of 0.79).
- Thresholds are **flagged `reliable=False`** below `reliable_windows` windows (default 30), and for a constant
  reference, a degenerate null or a saturated null. `options()` passes the flag to the drift results (`reliable` column), to mark their
  verdicts as provisional; `options(include_reliability=False)` leaves it out.
- `contiguous=True` (default) draws contiguous windows from a time-sorted reference, keeping the temporal dependence
  of real traffic; random rows (`contiguous=False`) would hide it and under-estimate false alarms.
- `options` keys `"metric|feature"` take precedence over per-metric keys in `Analyzer.run()`.
- `false_alarm_rate(metric, feature)` reports how often the nominal p < 0.05 rule fires on A/A draws: a direct measure
  of how misleading it would be on this traffic.
- `summary()` lists, per (metric, feature), the threshold and its interval (`threshold_interval()`), the status
  (`ok`, `constant_reference`, `degenerate_null`, `saturated`, `error`), the reliability, the p-value floor, the share
  of A/A draws at the bound of the statistic (`share_at_bound`) and the `between_within_ratio` of numeric features.
- A (metric, feature) that cannot be calibrated (e.g. a text column for `wasserstein`) is recorded in `errors` with a
  warning; the others are calibrated. The metrics' `missing_values` and `nan_policy` apply to the A/A draws.

**Number of draws and Benjamini-Hochberg.** An empirical p-value cannot go below `1 / (n_draws + 1)`, while
Benjamini-Hochberg needs `alpha / m` to reject one test out of `m`. With 200 draws (floor 0.005), a drift seen by three
tests of one feature could not pass the correction beyond about 30 tests per window: in the lab, detection fell from
100 % at 30 tests to 40 % at 33 and 0 % at 60, and only 8 of 100 models of 20 features with one drifted feature were
flagged. `fit()` therefore raises the number of draws until the floor is half of `fdr_alpha / m` (`fdr_alpha=0.05`,
`m` = `n_tests` or the number of metrics calibrated; `auto_draws=True`, capped by `max_draws=10000`): 1,599 draws for
40 tests (`n_draws_used`); an increase above 5 times `n_draws` is logged as a warning with its estimated cost. With
`auto_draws=False`, a warning is emitted instead. `empirical_pvalues()` and
`fdr_correct(..., pvalue_floor=calibrator.pvalue_floor)` warn when the floor is still above `alpha / m`.

**Reusing a calibration.** The calibration dominates the cost of an analysis (88 % of 4.9 s per model in the lab), and
the cost grows with the number of draws. The null distribution of a (metric, feature) depends only on the reference
values of the feature and on the draw settings, so it does not need to be computed again for every window:

```python
from pulsar_metrics.calibration import CalibrationCache

cache = CalibrationCache("calibration_cache")  # .npy files; CalibrationCache() keeps them in memory
calibrator = AACalibrator(window_size=300).fit(long_reference, monitor._metrics_list, cache=cache)
calibrator.save("credit_model_v1.json")  # or keep the whole calibration
calibrator = AACalibrator.load("credit_model_v1.json")
```

Cache entries are keyed by the metric and its missing-value settings, a hash of the feature's reference values (in
order), `window_size`, the number of draws, `contiguous`, `compare_to`, the level grouping and `seed`: two models that share a reference feature share
its entry, and a changed reference never reuses a stale one. The quantile is not in the key: the threshold is
recomputed from the stored values. With a `seed` that is not an integer, nothing is cached.

### Known limits of the calibration

- **Thresholds of short references vary.** On independent data with 10 windows, the calibrated `alpha` of `ks_2samp`
  ranged from 0.001 to 0.292 over 400 references (p10 0.020, p90 0.150, mean 0.080, nominal 0.05): about 45 disjoint
  window pairs feed the draws. The window-level false alarm rate after BH stayed at 0 %. Use 30 windows or more, read
  `threshold_interval()`, or set `shrink_draws` (e.g. 45) to mix that many pseudo-draws of the nominal alpha into the
  calibrated alphas. Shrinking gives back part of the protection against dependence, and distances are not shrunk.
- **Cost of the draws.** The number of draws grows with the number of tests (1,599 for 40), and with the default
  `compare_to="reference"` each draw runs on the whole reference: on 8,000 rows, about 7 times the cost of
  `compare_to="window"` (14.5 s against 2.0 s for 300 draws of `ks_2samp` and `wasserstein`, busy machine).
  On numeric features the statistic runs on numpy arrays, but each scipy test still costs a fraction of a
  millisecond: cache the calibration.
- **Calibration is per reference.** A static reference flags lasting changes (seasons, growth) as drift, which is
  correct; choosing and refreshing the reference is up to the user.
- **Saturated nulls are blind.** When the reference mixes regimes (e.g. a macro variable constant within each month),
  many A/A draws already reach the bound of the statistic: the `ks_2samp` p-value underflows to 0 and `psi` reaches
  its epsilon ceiling. A crisis window then ties with the null and is never flagged (lab TC-04: ks p = 0 in 47 % of the
  draws, calibrated alpha 0.0; psi at 27.63 in 44 %; BH detection 0 % against 65 to 100 % for `wasserstein`). When more
  than 5 % of the draws sit at the bound, the null gets `status="saturated"`, `reliable=False` and a warning, and the
  drift results carry the status. It is a warning, not a fix (a strict inequality on ties turns the blindness into
  false alarms): use `wasserstein`, which is unbounded, for such features, or a reference per regime. Ties at p = 1 are
  on the non-critical side and do not count.
- **Seasonal references absorb the season.** A/A draws on a whole year compare winter with summer weeks, so the
  thresholds swallow a change of that size (lab TS-04, bike rentals +63 % in 2012: confirmed in 0 % of the weeks with
  all of 2011 as reference, 76 % with the same weeks of 2011). `fit()` reports, per numeric feature, the
  `between_within_ratio` (window size x variance of the window means / mean within-window variance: about 1 for
  stationary data, (1 + phi) / (1 - phi) for AR(1) traffic) and warns above 10. Calibrate on, and compare with, the
  same period of the reference:

```python
from pulsar_metrics.calibration import CalibrationCache, period_reference

cache = CalibrationCache()
ref = period_reference(reference, window, weeks=4)  # same weeks of the year +- 4 weeks, every year of the reference
calibrator = AACalibrator(window_size=len(window), allow_short=True).fit(ref, monitor._metrics_list, cache=cache)
monitor.run(current=window, reference=ref, options=calibrator.options())
```

  A period reference holds few windows (9 to 11 weeks for one year): its thresholds are flagged `reliable=False`; use
  several years when available.

## Confirming drift over successive windows

In continuous monitoring, a single window can flag a drift by chance: at a 5 % significance level, a statistical test
raises a false alarm about once every twenty windows on stable data. `DriftPersistence` confirms a drift only when it is
seen in `windows` consecutive analyses (default 2), and clears it only after as many analyses without drift.

```python
from pulsar_metrics.analyzers.persistence import DriftPersistence

persistence = DriftPersistence(windows=2)  # keep one instance for the whole monitoring

for shift in [0.0, 0.8, 0.0, 0.8, 0.8]:  # stable, isolated shift, stable, lasting shift
    window_data = window(300, shift, "2026-03-01")  # helper from the quick start
    window_analyzer = Analyzer(name="window", model_id="credit_model", model_version="1.0")
    window_analyzer.add_drift_metrics(metrics_list=["ks_2samp"], features_list=["income"])
    window_analyzer.run(current=window_data, reference=reference)

    verdict = persistence.update_to_pandas(window_analyzer.results_to_pandas()).iloc[0]
    print(f"shift={shift}  raw={verdict.drift_status}  confirmed={verdict.confirmed_drift}  changed={verdict.changed}")
```

```
shift=0.0  raw=False  confirmed=False  changed=False
shift=0.8  raw=True  confirmed=False  changed=False    <- isolated: not confirmed
shift=0.0  raw=False  confirmed=False  changed=False
shift=0.8  raw=True  confirmed=False  changed=False
shift=0.8  raw=True  confirmed=True  changed=True     <- second window in a row: confirmed
```

Rules, for each (metric type, metric name, feature): `drift_status=True` extends the drift streak, `False` extends the
no-drift streak, and `None` (no threshold, metric in error) keeps both. `changed=True` marks the window where the
confirmed verdict switches, which is the event to act upon (alert, re-evaluation). `confirmed()` lists the metrics in
confirmed drift and `reset()` clears the history, for instance after deploying a new model version. Summary statistics
are ignored.

The cost is a delay of `windows - 1` windows before a lasting drift is confirmed.

## Data quality against a reference profile

Drift tests compare distributions; they do not say that a column disappeared, that a sensor now reports in another
unit or that a batch was loaded twice. `pulsar_metrics.quality` learns a **data-quality profile** on a reference
window, versioned with the model, and checks every later window against it, with thresholds set per check and per
column in a policy.

```python
from pulsar_metrics.quality import DataQualityAnalyzer, DataQualityProfile

profile = DataQualityProfile.fit(
    reference,  # e.g. the training window
    rules=["dew <= temp", "rh <= 100"],  # optional row rules: column <op> column | number
    reference_id="weather-2016-2018",
)
profile.save("quality_profile.yaml")  # or .json; DataQualityProfile.load() reads it back

analyzer = DataQualityAnalyzer(
    "weekly quality", model_id="no2", model_version="1",
    profile=DataQualityProfile.load("quality_profile.yaml"),
    policy="quality_policy.yaml",  # or a dict, or None for the defaults
)
analyzer.run(current)  # or run(current, reference) to learn the profile on the first call
print(analyzer.results_to_pandas()[["metric_name", "feature_name", "metric_value", "threshold", "drift_status", "status"]])
```

The profile holds aggregates only: per column its kind (`numeric`, `categorical`, `boolean`, `datetime`) and dtype, its
share of missing values, its cardinality, its min, max and quantiles (numeric) and its levels (categorical and boolean,
up to `max_categories=50`: identifiers and free text keep no domain); for the frame, the share of duplicate rows (on
every profiled column, or on a key given as `duplicate_columns`); and the share of reference rows breaking each rule.
`model_id`, `model_version` and `pred_timestamp` are left out (`exclude=`); when they are present and the analyzer has
a `model_id`, rows are filtered on them as in `Analyzer`. YAML needs PyYAML.

Each check gives one `MetricResults` per column (`metric_type="data_quality"`), with the share in `metric_value`, the
threshold of the policy in `threshold` and `drift_status=True` when the share is **above** it:

| `metric_name` | `feature_name` | Value | Default threshold |
|---|---|---|---|
| `schema` | column | 1 when the column is missing from the window (`status="missing_column"`), new (`"new_column"`, unless `allow_new_columns`) or of another kind (`"type_change"`), else 0 | 0 |
| `null_share` | column | Share of missing values | reference share + 0.05 |
| `out_of_range_share` | numeric column | Share of the values outside the reference min-max (or `range_quantiles`); missing values set aside | 0.01 (+ the reference share outside the quantiles) |
| `unseen_category_share` | categorical or boolean column | Share of the values whose level is not in the reference; missing values set aside | 0.01 |
| `duplicate_share` | `None` (frame) | Share of rows repeating an earlier row | reference share + 0.01 |
| `rule_violation_share` | rule id | Share of the rows (both sides present) breaking the rule | reference share + 0.01 |

The kind is read on the non-missing values: an integer or boolean column that gets missing values is not a type
change. Results report counts and shares only, never values of the window (an unseen level may be personal data typed
in the wrong field).

**Policy.** A threshold is absolute (`0.1` or `{max: 0.1}`) or relative to the reference share (`{margin: 0.05}`);
`null` disables a check. Top-level entries apply to every column, `columns` overrides them per column (or per rule id):

```yaml
null_share: {margin: 0.05}
out_of_range_share: 0.01
range_quantiles: null        # [0.001, 0.999]: range on reference quantiles instead of its min and max
allow_new_columns: false
columns:
  weather:                   # empty when no phenomenon is reported, more often in summer
    null_share: {margin: 0.25}
    unseen_category_share: 0.03
  station_id: {unseen_category_share: null, schema: false}
```

Unknown keys and shares outside [0, 1] are refused. **A check that cannot run** has no value and no verdict: `status`
says why and `reliable` is `False`: `empty_window` (an empty window is recorded, not refused; the schema check still
runs), `missing_in_window`, `not_in_reference` (a new column), `type_mismatch` (range of a column no longer numeric),
`no_reference_values`, `no_reference_domain` (more levels than `max_categories`), `no_values_in_window`.
`pulsar_metrics.quality.check(profile, current, policy)` runs the checks without an analyzer.

On the hourly weather of Montréal-Trudeau airport (open data of Environment and Climate Change Canada, input of the
NO2 trial of [pulsar_demo](https://github.com/Rocket-Science-Development/pulsar_demo)), with a 2016-2018 profile and the
policy above, the 36 months of 2019-2021 raise no alert, while July 2019 with the pressure in hPa instead of kPa, a
third of the humidity missing, two weather labels in French, a day loaded twice, dew point and temperature swapped or
humidity sent as text each raise the alert of its check (`tests/test_data_quality_gc.py`, skipped without the raw
files). With the default thresholds alone, 16 of the 36 months raise an alert, all on `weather`: its share of empty
values is seasonal (52 % in the reference, up to 65 % in summer: 15 months), and one winter month brings new
combinations of levels (1.5 % of its values), hence the per-column entries.

`to_evidence(analyzer, system_id)` exports the results as evidence records of family `data_quality`, `direction`
`upper` and `decision_basis` `value`; checks that cannot run are `not_evaluable` with their status as reason code.

## Evidence records (stable export format)

`MetricResults` is the in-memory result of an analysis. To hand results over to another system (an audit trail, a
central monitoring or governance service) without that system depending on pulsar-metrics, `pulsar_metrics.evidence`
turns them into **evidence records**: one finding about one monitored system, in a versioned, typed and
self-describing JSON format, `"pulsar.evidence/1"`, described by a JSON Schema shipped with the package
(`pulsar_metrics/schemas/evidence-1.json`).

```python
from pulsar_metrics.evidence import to_evidence

records = to_evidence(
    analyzer,  # after run(); also a DataQualityAnalyzer, a FairnessAnalyzer, one MetricResults or a list of them
    system_id="SYS-0042",  # your identifier of the monitored system (e.g. its entry in an inventory of AI systems)
    reference_id="holdout-2026-01",
    calibrator=calibrator,  # optional: AACalibrator behind the thresholds
    persistence=persistence,  # optional: DriftPersistence after this window
)
payload = [r.to_dict() for r in records]  # JSON-ready; Evidence.from_json / from_dict read them back
```

| Field | Meaning |
|---|---|
| `schema_version` | `"pulsar.evidence/1"` |
| `evidence_id` | sha256 of the identifying content (system, model, metric family and name, feature, group, subject metric, window, reference, and the rules version of compliance records): the same finding sent twice has the same id, so a receiver can deduplicate; a recomputation of the same window keeps the id (keep the latest `produced_at`). A window with no `start`, `end` nor `id` cannot tell two windows apart: `produced_at` is then hashed too and the converters warn, so pass `window={"id": ...}` (batch, run or period id) for results without timestamps (fairness, compliance, GenAI tests). `from_dict` / `from_json` recompute the id and refuse a record whose id does not match its content |
| `system_id`, `model_id`, `model_version` | Monitored system (identifier supplied by the user) and model |
| `metric` | `family` (`drift`, `performance`, `fairness`, `genai`, `compliance`, `data_quality`, `explainability`, `custom`), `name`, `feature`, `group` / `reference_group` (fairness), `subject_metric`, `value`, `threshold` (number or `[low, high]`, a `null` bound being unbounded), `direction` (`upper`: alert above, `lower`: alert below, `outside`: alert outside the zone, `none`), `decision_basis` (`value`, `p_value`, `confidence_interval`, `reference_test`, `sequential_test`), `p_value`, `ci_low` / `ci_high`, `e_value`, `n`, `status`, `error`, `method` |
| `metric.status` | `ok`, `alert` (drift, degradation, disparity or violation detected), `no_verdict` (no threshold, inconclusive), `not_evaluable` (suppressed group, no ground truth, rule not evaluable), `error` |
| `window`, `reference` | `id` (yours, optional), `start`, `end` (UTC, ISO 8601; NaT is read as missing), `n_rows` |
| `reliability` | `reliable` (bool), `reasons` (codes that make it false), `notes` (codes that lower the confidence only), `reason_text` (Pulsar's message), `calibration` (`method`: `aa_calibration`, `nominal` or `none`; `null_status`, `window_size`, `quantile`, reference windows, draws, share at bound, 90 % interval of the threshold, empirical p-value, p-value floor), `persistence` (`windows`, `drift_streak`, `no_drift_streak`, `confirmed`, `changed`), `quality` (missing shares, `low_expected_share`, `n_labelled`, `label_coverage`, `suppressed`) |
| `provenance` | `producer` (`pulsar-metrics`), `producer_version`, `produced_at` (UTC), `run_name` |
| `details` | Free aggregates (e.g. `conf_int`, `reference_value` of summary statistics, `pulsar_status` of a schema check: `missing_column`, `new_column`, `type_change`). **Aggregates only, never raw rows, texts or personal data**; capped at 64 KiB |

Reliability codes (`REASON_CODES`, `NOTE_CODES` in the module):

- `reasons`, `reliable` is then `false`: `computation_error`, `no_ground_truth`, `insufficient_labels`,
  `few_labelled_rows`, `constant_reference`, `saturated_null`, `degenerate_null`, `window_size_mismatch`,
  `short_reference`, `suppressed_group`, `rule_not_evaluable`, `instrument_unusable`, `flagged_unreliable`,
  `non_finite_value` (NaN or infinite value, exported as `null`), `unstable_reference` (explainability),
  `few_repeats` and `few_questions` (consistency), and for the data-quality checks that cannot run `empty_window`,
  `missing_in_window`, `not_in_reference`, `type_mismatch`, `no_reference_values`, `no_reference_domain`,
  `no_values_in_window`;
- `notes`: `nominal_threshold`, `no_threshold`, `sparse_cells`, `missing_values_set_aside`, `partial_label_coverage`,
  `drift_not_confirmed`, `ecological_inference`, `small_canary`.

`reliable` is stricter than `MetricResults.reliable`: it is also false for a failed or not evaluable metric and for a
constant reference. `metric.status` is the verdict of the window itself; with `persistence`, `confirmed` says whether
the alert has lasted (`drift_not_confirmed` otherwise). Records are strict JSON: NaN and infinities are refused by
the models, `loads()` and `dumps()`.

**Compatibility.** Additive changes keep `"pulsar.evidence/1"`: a new optional field, a new code or family, a new key
of `details`. Readers must ignore unknown fields and treat unknown codes as unknown (`Evidence.from_json` does). Any
other change (field removed, renamed or retyped, meaning of a value changed, identity fields of `evidence_id` changed)
is breaking and moves to `"pulsar.evidence/2"`. The committed schema is checked against the model by the tests;
regenerate it with `python -c "from pulsar_metrics.evidence import write_json_schema; write_json_schema()"`.

To send the records over HTTPS with local buffering and retries, see `EvidenceOutbox` in
[pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection).

## Monitoring LLM applications and agents

[pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection) turns OpenTelemetry
GenAI traces into one `agent_run` record per agent run (`steps`, `tool_calls`, `first_tool`, `tool_error_rate`,
`repeated_calls`, `out_of_scope_calls`, `input_tokens`, `latency_s`, `termination`...). These records are ordinary
features: categorical ones (which tool, termination type) go to `chi2`, numeric ones (steps, tokens, latency) to
`ks_2samp` or `wasserstein`. Agents are non-deterministic, so calibrate the thresholds on A/A draws and confirm drifts
over several windows.

```python
import numpy as np
import pandas as pd

from pulsar_metrics.analyzers.base import Analyzer
from pulsar_metrics.calibration import AACalibrator
from pulsar_metrics.multiple_testing import fdr_correct

rng = np.random.default_rng(1)
TOOLS = ["search_programs", "check_eligibility", "compute_amount", "create_file"]


def agent_runs(n, start, tool_fault=0.0):
    """agent_run records as written by pulsar-data-collection (synthetic here)"""
    failed = rng.random(n) < tool_fault  # a degraded tool makes the agent retry
    return pd.DataFrame(
        {
            "model_id": "permit_assistant",
            "model_version": "1.0",
            "pred_timestamp": pd.date_range(start, periods=n, freq="min"),
            "first_tool": rng.choice(TOOLS, size=n, p=[0.2, 0.35, 0.3, 0.15]),
            "steps": rng.poisson(2, n) + 1 + 2 * failed,
            "tool_error_rate": np.where(failed, 0.5, 0.0),
            "latency_s": rng.gamma(4, 5, n) + 20 * failed,
        }
    )


agent_reference = agent_runs(3000, "2026-11-01")  # stable traffic, at least 10 windows
agent_current = agent_runs(200, "2026-11-04", tool_fault=0.3)  # one window, 30 % of runs hit a failing tool

agents = Analyzer(name="agent", model_id="permit_assistant", model_version="1.0")
agents.add_drift_metrics(metrics_list=["chi2"], features_list=["first_tool"])
agents.add_drift_metrics(metrics_list=["ks_2samp"], features_list=["steps", "latency_s"])
agents.add_drift_metrics(metrics_list=["wasserstein"], features_list=["tool_error_rate"])

calibrator = AACalibrator(window_size=200, n_draws=200).fit(agent_reference, agents._metrics_list)
agents.run(current=agent_current, reference=agent_reference, options=calibrator.options())

drift = agents.results_to_pandas().query("metric_type == 'drift'").copy()
drift["p_empirical"] = calibrator.empirical_pvalues(drift)
print(fdr_correct(drift, "p_empirical")[["metric_name", "feature_name", "p_adjusted", "drift_fdr"]].to_string(index=False))
```

```
metric_name    feature_name  p_adjusted drift_fdr
       chi2      first_tool    0.880597     False
   ks_2samp           steps    0.006633      True
   ks_2samp       latency_s    0.006633      True
wasserstein tool_error_rate    0.006633      True
```

The tool distribution (`first_tool`) is unchanged, while steps, latency and tool errors drift: the signature of a
degraded tool rather than of a change in the requests. Feed each window's results to `DriftPersistence` to raise an
alert only when the drift lasts.

### Abstentions and answers without support (RAG agents)

pulsar-data-collection records two 0 / 1 indicators per run of a retrieval agent (`genai.abstention`):
`det_abstention` (the answer says the information is not found) and `det_unsupported_answer` (the agent answered
although the retrieval was empty or irrelevant, or the answer is not supported by the passages). Their rates per
window are tested with `rate_increase`, a one-sided Fisher exact test (alert only when the window rate is higher):

```python
from pulsar_metrics.evidence import to_evidence
from pulsar_metrics.metrics.drift import DriftTestMetric

result = DriftTestMetric("rate_increase", "det_unsupported_answer").evaluate(window_runs, reference_runs)
print(result.metric_value, result.drift_status, result.method)
# 0.00051 True one-sided Fisher exact test, window rate higher: window 3/8 = 0.375 vs reference 0/80 = 0
record = to_evidence([result], "register-assistant", window={"id": "2026-11-04T10"})[0]
print(record.details)  # current_count, current_n, current_rate, reference_count, reference_n, reference_rate
```

Why not the generic tests on these fields: a reference without any positive (a zero base rate, the usual case of
unsupported answers) makes every generic test mark the feature `constant_reference` (`reliable=False` in the
evidence), and the rank and chi2 tests are asymptotic: on the window above, `manwu` (one-sided) gives 1.7e-08 where
the exact p-value is 5.1e-04. `rate_increase` takes 0 / 1 or boolean values only (anything else raises), and
`AACalibrator` does not flag its constant references either.

## Group fairness

`pulsar_metrics.fairness` compares a decision rule across groups defined by protected attributes. It follows what the
literature identifies as the weak points of fairness tooling: small groups, multiple groups and verdicts based on point
estimates. Every result carries the group size and a confidence interval, and **a disparity is flagged only when the
interval lies entirely outside the acceptable zone**.

| Criterion | Compares (group vs reference group) | Default acceptable zone | Needs |
|---|---|---|---|
| `disparate_impact_ratio` | selection rate ratio (4/5 rule) | [0.8, 1.25] | decisions |
| `demographic_parity_difference` | selection rate difference | [-0.1, 0.1] | decisions |
| `equal_opportunity_difference` | true positive rate difference | [-0.1, 0.1] | labels |
| `false_positive_rate_difference` | false positive rate difference | [-0.1, 0.1] | labels |
| `equalized_odds_difference` | max of the TPR and FPR differences (absolute) | [0, 0.1] | labels |
| `calibration_difference` | expected calibration error difference | [-0.05, 0.05] | labels and scores |
| `error_rate_difference` | mean absolute error difference (regression) | set it in the policy | labels |

- **Prediction-based criteria are the default.** Label-based ones must be chosen explicitly: labels arrive late and
  carry the biases of past decisions. Calibration and equalized odds cannot both hold when base rates differ: the
  choice belongs to the user (and to the impact assessment), not to the tool.
- **Verdict** (`drift_status`): `True` = disparity established (interval outside the zone), `False` = within the zone,
  `None` = inconclusive (the interval straddles a bound, typically a small group).
- **Intervals**: stratified bootstrap (each group resampled within itself; exact binomial replicates for 0/1
  decisions, vectorized); with several groups, a **simultaneous** max-t band so that the confidence level holds for
  all groups at once. Groups under `small_group_size` (200) rows get **score intervals** on the rate criteria
  (`interval="auto"`: Wilson combined by Newcombe for differences, MOVER on the log scale for the ratio), which
  keep their level
  where the percentile bootstrap does not; `interval="bootstrap"` restores the former behaviour.
- **Small cells are suppressed** (`min_group_size`, default 30): no value and no count are published for them.
  `coverage()` reports, per feature and criterion, the share of rows in suppressed or inconclusive groups
  (`no_verdict_share`), and a warning is raised when more than `max_no_verdict_share` (5 %) of the rows are in
  suppressed cells.
- **Intersections**: `intersections=True` also evaluates combinations of attributes (`sex=F|region=North`), up to
  `max_intersection_depth` attributes (default 2; `None` for all of them, `max_cells` to cap the number of cells).
  Combinations without any publishable cell are skipped with a warning.
- **Inconclusive verdicts say how much data would settle them**: `method` ends with the number of rows of the group
  needed for an 80 % chance of a verdict at the observed rates (`required_group_size` in
  `pulsar_metrics.fairness`).
- **Equivalence (TOST)**: the default "within the zone" verdict is a TOST at 2.5 % per side;
  `equivalence_confidence=0.90` gives the usual 5 % per side, while a disparity is still established at
  `confidence`.
- **Pre-declared cells**: `declared_groups=["sex=F", ...]` gives verdicts only to the declared cells, under a
  narrower simultaneous band; the other cells are reported as exploratory (`drift_status` None).
- **Missing attribute values** (NaN, None) form a group of their own, `<missing>` (e.g. `sex=<missing>`), judged
  and suppressed like any other: rows whose attribute is unknown are often the ones a report must not drop
  silently. Recode them beforehand to exclude them.
- **Protected attributes do not need to travel with the predictions**: pass them separately with a pseudonymous id;
  the join happens in memory and only group-level results come out. Several attribute rows for one id without
  `share=` / `count=` are refused with an explicit message (previously a raw pandas `MergeError`).
- **Numeric protected attributes**: a numeric attribute with more than `numeric_levels` (20) distinct values raises
  a warning, and the error raised when every one of its groups is suppressed suggests binning. `bins={"share": 3}`
  bins it by quantiles (groups `Q1 [low, high]`, `Q2 (low, high]`, ...; `bins={"share": [0, 0.05, 1]}` for explicit
  edges, groups `B1`, `B2`, ...), the edges are kept in `bin_edges` and in `method`, and `reference_groups={"share":
  "Q1"}` names the reference by its prefix.

```python
from pulsar_metrics.fairness import FairnessAnalyzer, FairnessPolicy

rng = np.random.default_rng(7)
n = 4000
applicants = pd.DataFrame({"applicant_id": [f"a{i}" for i in range(n)], "region": rng.choice(["North", "South"], n)})
applicants["language"] = rng.choice(["fr", "en"], n)
# decisions: the rule approves 50 % of English-speaking applicants but 35 % of French-speaking ones
approved = rng.random(n) < np.where(applicants["language"] == "fr", 0.35, 0.5)
predictions = pd.DataFrame({"applicant_id": applicants["applicant_id"], "y_pred": approved.astype(int)})

fairness = FairnessAnalyzer(
    ["language", "region"],
    FairnessPolicy(criteria=["disparate_impact_ratio", "demographic_parity_difference"], min_group_size=30),
    reference_groups={"language": "en"},
    intersections=True,
)
fairness.run(predictions, attributes_data=applicants, on="applicant_id")
report = fairness.results_to_pandas().query("metric_name == 'disparate_impact_ratio'")
print(report[["feature_name", "group", "reference_group", "n", "metric_value", "ci_low", "ci_high", "drift_status"]]
      .round(3).to_string(index=False))
```

```
   feature_name                    group          reference_group    n  metric_value  ci_low  ci_high  drift_status
       language              language=fr              language=en 2023         0.701   0.649    0.753          True
         region             region=South             region=North 1999         1.011   0.944    1.096         False
language|region language=fr|region=South language=fr|region=North 1007         1.026   0.880    1.173         False
language|region language=en|region=South language=fr|region=North  992         1.442   1.257    1.627          True
language|region language=en|region=North language=fr|region=North  985         1.446   1.271    1.622          True
```

Validated against [Fairlearn](https://fairlearn.org) on ACSIncome (Folktables, Massachusetts 2018, 12,035 test
rows): identical disparate impact ratio, equal opportunity and equalized odds differences for sex (difference 0).

**Planning the data a verdict needs** (`pulsar_metrics.fairness`): `required_group_size` (rows of a group),
`paired_sets_needed` and `flip_sets_needed` (counterfactual probe sets), `sequential_decisions_needed` (horizon of
`FairnessMonitor`). Each takes the expected rates and returns the size for a conclusive verdict with probability
`power` (0.8), or `None` when no size suffices (disparity on a bound of the zone).

```python
from pulsar_metrics.fairness import required_group_size

print(required_group_size(0.5, 0.5, zone=(-0.1, 0.1), reference_size=2000))  # rows to certify a fair group
```

```
299
```

#### Aggregate data: weights, composition tables and units

`FairnessAnalyzer` assumes by default that a row is a person. When rows are units (communities, branches, schools)
and the protected attribute is only known as a composition (the share of each group in the unit):

- `weights="population"` weights every group rate and every interval by a row weight. Score intervals and the
  binomial shortcut, which count independent rows, give way to the bootstrap.
- a **composition table** passed as `attributes_data`, several rows per unit with the share (`share="share"`) or the
  count of people (`count="residents"`) of each group, splits each unit's decision across its groups. With
  `weights="population"` and shares, a group's selection rate is the share of its people who live in a flagged unit
  (population-weighted exposure).
- `cluster="unit_id"` resamples units, not rows, jointly across groups. `cluster` defaults to `on` with a
  composition table. `n` and `min_group_size` then count units: **the effective sample size is the number of units**,
  however many rows (or people) each unit has. Expanding aggregates into pseudo-individual rows without `cluster=`
  gives intervals that shrink as 1/sqrt(copies), an arbitrary choice; a warning is raised when more than half of
  the rows repeat another row exactly.
- results built from a composition carry `status="ecological"`. They describe the people of each group through the
  decisions taken on their units, not decisions taken on individuals. `unit="community"` names the unit in `method`.

```python
fairness = FairnessAnalyzer(["race"], FairnessPolicy(criteria=["demographic_parity_difference"]),
                            reference_groups={"race": "White"}, weights="population", unit="community")
# predictions: one row per community (cid, y_pred, population); composition: cid, race, share (several rows per cid)
fairness.run(predictions, attributes_data=composition, on="cid", share="share")
```

On the lab data (UCI Communities and Crime, 1,994 communities, FA-03), this gives the population-weighted gap
computed by hand in the lab: Black - White +0.376 [+0.335, +0.415] and Asian - White +0.128 [+0.048, +0.183],
inconclusive, against the lab's cluster bootstrap [+0.334, +0.415] and [+0.044, +0.183]. On the same data expanded
to 200,000 and 2,000,000 pseudo-individual rows, the Asian - White interval was 0.023 and 0.007 wide, with an
"established" verdict. With `cluster="community"` it is 0.164 and 0.160 wide, and the verdict stays inconclusive.

#### Known limits (night lab, FA-01 and FA-04)

| Limit | Lab figure | What the code does now |
|---|---|---|
| Intersections explode with the number of attributes | 5 attributes: 1,146 to 1,750 cells, 65 to 75 % suppressed, 4 to 11 % of the rows without verdict, 30 to 80 times the run time | pairs by default (`max_intersection_depth=2`), `max_cells`, `coverage()` and a warning; exact binomial bootstrap (5 attributes x 50,000 rows, all combinations: 188 s before, 5 s now) |
| The simultaneous band is wide | 2.2 times the per-group intervals over 609 cells, 176 inconclusive verdicts instead of 81 | `declared_groups`; the trade-off itself is the price of a level that holds for every cell |
| Small groups get no verdict | groups under 300 rows: 152 of 199 parity results inconclusive | the rows needed are reported; pooling years or regions is left to the user and should be declared in the analysis |
| The percentile bootstrap is anti-conservative for small groups | one-sided misses 3 to 5 % instead of 2.5 % (30 to 100 rows) | score intervals under 200 rows (simulated: at most 3.3 % per side, about 5 % in total) |
| Small fair groups cannot be certified | no bias, zone +-0.1: certified 0 / 0 / 5 / 52 / 96 % of the time at 30 / 50 / 100 / 200 / 500 rows | `equivalence_confidence=0.90` (simulated at 100 / 200 rows: 29 / 70 % certified instead of 2 / 52 %); below about 100 rows no method certifies a zone of +-0.1: a Wilson interval at 30 rows is +-0.17 wide |

### Continuous fairness monitoring (anytime-valid)

Re-running the analysis above after every batch inflates false alarms: with enough looks, a test at 5 % ends up
firing on a fair system. `FairnessMonitor` watches a stream of decisions with **sequential tests by betting** (Chugg
et al., NeurIPS 2023): the probability that it *ever* raises a false alarm, however often it is looked at, stays below
`alpha` as long as every group stays within the tolerance. A rejection is final and reports the decision at which the
disparity was established and its e-value (evidence, ≥ 1 / alpha).

- One test per monitored group against the reference group; `alpha` is split over the groups (Bonferroni).
- Criteria: `demographic_parity` (selection rates) and `equal_opportunity` (true positive rates, needs `y_true`).
- `tolerance`: absolute difference of rates considered acceptable (default 0.05); the test asks whether the gap
  exceeds it, not whether it is exactly zero.
- **Level or change**: by default the test asks whether the gap exceeds the tolerance. A gap that was already there
  when monitoring started alarms at once and stays alarmed, which hides any later degradation. `fit_baseline(frame)`
  (or `baseline={"fr": -0.13}`) switches to the change mode: H0 |gap - baseline gap| <= tolerance.
- **Re-arming**: `acknowledge(group, rebaseline=None)` starts a new test for the group once an alarm has been handled
  (the alarm is kept in `alarms`); each armed period has its own alpha guarantee.
- **Reference pooling** (`reference_pooling=True`, default): each group decision is paired with the mean of a
  disjoint share of the batch's reference decisions, so no reference decision is wasted (`False`: the 1:1 pairs of
  0.10 and before, which dropped about 95 % of the reference decisions for a group of 5 %).
- Batches can have any size and any group mix; unpaired group observations wait for the next batch.

```python
from pulsar_metrics.fairness import FairnessMonitor

monitor = FairnessMonitor("language", reference_group="en", groups=["fr"], tolerance=0.05, alpha=0.05)
rng = np.random.default_rng(1)
for day in range(30):
    language = rng.choice(["fr", "en"], 400)
    # from day 10 on, French-speaking applicants are approved 35 % of the time instead of 50 %
    rate = np.where((language == "fr") & (day >= 10), 0.35, 0.5)
    batch = pd.DataFrame({"language": language, "y_pred": (rng.random(400) < rate).astype(int)})
    result = monitor.update(batch)[0]
    if result.drift_status:
        print(f"day {day}: disparity established after {result.n} pairs, e-value {result.e_value:.0f}")
        break
```

```
day 15: disparity established after 3138 pairs, e-value 22
```

A bias-injection bench (`benchmarks/fairness_drift.py`, 1,500 simulations per source, tolerance 0.05, alpha 0.05)
measures the trade-off against a fixed-sample test re-run after every batch:

| | Monitor (betting) | Fixed test re-run after each batch |
|---|---|---|
| False alarms, no disparity, 22,000 decisions (synthetic) | 0 – 1 % | 0 – 4 % |
| False alarms, real gap just within the tolerance (ACSIncome, Folktables) | 1 – 2 % | 4 – 10 % |
| Power, gap 0.09 beyond the tolerance, group ≥ 20 % of the stream | ≥ 0.99 | ≥ 0.94 (the others alarmed falsely before the injection) |
| Median delay, synthetic, group 20 %, rate lowered by 0.10 | 10,000 decisions | 6,000 decisions |

The guarantee costs detection delay (typically 1 to 3 times longer), most visibly for small groups and gaps barely
beyond the tolerance. The bench above used the 1:1 pairs.

#### Known limits (night lab, FA-01 and FA-04)

- **Small groups**: a drop of 0.1 on a group of 5 % of the stream (tolerance 0.05, 20,000 decisions) was established
  in 22 % of the lab's runs with 1:1 pairs. On the same setting, simulated with 200 runs, reference pooling raises it
  from 16 % to 52 % (median delay 12,000 decisions), and from 85 % to 98 % for a group of 20 %, with 0 to 0.5 % false
  alarms. `sequential_decisions_needed(gap, tolerance, group_share)` gives the order of magnitude of the horizon
  (about 15,500 decisions pooled, 29,500 with 1:1 pairs, for the case above).
- **Level mode on stable gaps**: in the lab, 23 of 23 gaps that were stable but beyond the tolerance alarmed from the
  first year and never re-armed. Use the change mode and `acknowledge` for them.
- The baseline of the change mode is a point estimate; the test takes it as given, so choose a tolerance above its
  uncertainty (`baseline_n` keeps the rows behind each gap).

### Counterfactual probes of LLM applications and agents

`counterfactual_analysis` and `comparative_analysis` analyse the rows collected by the counterfactual probes of
`pulsar_data_collection.genai.probes`: requests that differ only by one attribute (e.g. a first name typical of each
group). They answer whether that attribute changes **what the application does** (the answer, the tool it calls, a
request for authentication, a refusal), and they judge it against controls rather than against zero:

| Result | Meaning | Verdict |
|---|---|---|
| `placebo_flip_rate` | share of placebo sets (an attribute that must not matter) whose outcome changes: noise floor of the instrument | — |
| `counterfactual_flip_rate` | share of sets whose outcome changes when only the attribute changes (Wilson interval) | — |
| `counterfactual_excess_flip_rate` | flip rate minus placebo flip rate (Newcombe interval) | above `flip_tolerance` (0.05) |
| `counterfactual_action_rate_difference` | rate of a given action in the group minus in the reference group, paired by set (BCa bootstrap); McNemar exact test, Benjamini-Hochberg over groups, in `method` | outside `rate_zone` |
| `counterfactual_action_rate_ratio` | same rates, ratio (4/5 rule) | outside `ratio_zone` |
| `unusable_rate` | comparative probes: share of the answers excluded from the analysis (call errors, refusals, unparsed answers), overall and per pair of groups when there are several; `method` gives the count of each kind and the sets still usable in both orders | above `max_unusable_rate` (0.5) |
| `position_effect` | comparative probes: share of answers picking the first profile; `True` = the answers depend on the order | binomial test |
| `comparative_instrument` | `"usable"` or `"instrument_unusable"` (also in `status`): unusable when more than `max_unusable_rate` of the answers are unusable, or when the answers depend on the order and more than `max_order_driven_share` (0.5) of the sets decided in both orders chose the same position twice | — |
| `tie_rate`, `comparative_preference` | share of ties; probability of choosing a group, averaged over both orders | outside `preference_zone`; when the answers depend on the order, only a preference strong enough to show through is reported, never its absence |

```python
from pulsar_metrics.fairness import counterfactual_analysis

# rows of a CounterfactualProbe run: the outcome is the first tool the agent called
rng = np.random.default_rng(2)
rows = []
for s in range(200):
    u = rng.random()  # what the request itself calls for, shared by the set; 3 % of answers flip at random
    for variant, p_auth in (("fr_qc", 0.1), ("en_ca", 0.1), ("maghreb", 0.45)):  # the agent asks one group to authenticate
        auth = (u < p_auth) != (rng.random() < 0.03)
        rows.append({"set_id": f"cf-{s}", "variant": variant, "attribute": "name", "control": "none",
                     "outcome": "authenticate" if auth else "check_eligibility"})
    u = rng.random()
    for variant in ("ref_a", "ref_b"):  # placebo: the file reference changes, nothing else
        auth = (u < 0.1) != (rng.random() < 0.03)
        rows.append({"set_id": f"pl-{s}", "variant": variant, "attribute": "reference", "control": "placebo",
                     "outcome": "authenticate" if auth else "check_eligibility"})
results = counterfactual_analysis(pd.DataFrame(rows), reference_variant="fr_qc", action="authenticate")
for r in results:
    print(f"{r.metric_name:<38} {r.group or r.feature_name:<13} {r.metric_value:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] {r.drift_status}")
```

```
placebo_flip_rate                      reference     +0.055 [+0.031, +0.096] None
counterfactual_flip_rate               name=en_ca    +0.065 [+0.038, +0.108] None
counterfactual_excess_flip_rate        name=en_ca    +0.010 [-0.039, +0.059] None
counterfactual_flip_rate               name=maghreb  +0.365 [+0.301, +0.434] None
counterfactual_excess_flip_rate        name=maghreb  +0.310 [+0.234, +0.383] True
counterfactual_action_rate_difference  name=en_ca    +0.005 [-0.030, +0.040] False
counterfactual_action_rate_ratio       name=en_ca    +1.053 [+0.714, +1.500] None
counterfactual_action_rate_difference  name=maghreb  +0.345 [+0.280, +0.415] True
counterfactual_action_rate_ratio       name=maghreb  +4.632 [+3.185, +7.250] True
```

`equivalence_confidence=0.90` judges "within the zone" and "no excess flips" with a TOST at 5 % per side (the
default is 2.5 % per side). Inconclusive results end their `method` with the number of probe sets needed for an
80 % chance of a verdict at the observed rates (`paired_sets_needed`, `flip_sets_needed`).

An answer is unusable when its call failed (`error`), when `choice` is `"refusal"` or when `choice` is missing or
holds anything other than `"first"`, `"second"` or `"tie"` (unparsed). Unusable answers are excluded from
`position_effect`, `tie_rate` and the preferences, and reported in `unusable_rate`. With an unusable instrument,
the preferences carry `status="instrument_unusable"`: an established preference is kept, but no verdict of absence
is given.

#### Known limits (night lab, FA-04, LB-01 and LB-02)

- **The comparative probe answers the position, not the profiles, on small local models** (LB-02, French
  CrowS-Pairs, 6 models): the share of "first" answers ranged from 0.028 to 0.991. `comparative_instrument` flags all
  6 as unusable: 58 % (ministral-3:8b) to 98 % (llama3.1:8b) of the sets decided in both orders chose the same
  position twice. llama3.2:3b refused 23.3 % of its answers (326 of 1,400, all unparsed). Before, they were dropped
  silently; `unusable_rate` now reports them, with 513 of 700 sets usable in both orders.

- **30 probe sets rarely certify anything**: without any bias, the action rate difference was "within the zone"
  (+-0.1) in 10 % of runs at 30 sets, 79 % at 60 and 97 % at 100. With no discordant set at all, the bound on the
  difference is the Wilson upper bound of 0 out of 30 sets, 0.11, already beyond the zone. Plan 60 to 100 sets
  (`paired_sets_needed(0.0, 0.0)` gives 60).
- **Excess flip rates stay inconclusive at 200 sets** when they sit near the tolerance: 7 (model, group) pairs out
  of 12 in LB-01, e.g. 0.038 [0.005, 0.077] against a tolerance of 0.05. That is a power limit, not a defect: use
  `flip_sets_needed` to size the probe. An excess of 0.038 sits 0.012 from the tolerance: about 4,200 probe and
  placebo sets for an 80 % chance of a verdict (`flip_sets_needed(0.058, 0.02)`); an excess of 0.015 needs about 400.

### Text drift, model equality and agent compliance

`pulsar_metrics.genai` adds these tests for LLM applications and agents:

| Function | Question | Test |
|---|---|---|
| `embedding_drift` / `mmd_test` | Has the **content** of the requests (or answers, or retrieved passages) changed, beyond what lengths and tools show? | MMD between the embeddings of two windows, Gaussian mixture kernel (0.5 to 4 times the median distance, or a fixed `bandwidth`), PCA fitted on both windows pooled, permutation p-value |
| `component_test` (`embedding_drift(test="component")`) | Has the **mix** of texts moved along a few directions (more positive reviews, more questions of one kind)? | Shift of the mean PCA components: Hotelling statistic with a ridge (`statistic="hotelling"`), or largest Welch t over the components (`"max_t"`, which names the component); permutation p-value |
| `novelty_test` (`embedding_drift(test="novelty")`) | Has a **small share of texts unlike the reference** appeared (another language, spam, an injection campaign)? | Distance of each text to its k nearest reference texts, threshold from a held-out part of the reference; share of novel texts against the expected share, permutation p-value; indices of the novel texts returned |
| `detection_power` | Which change can a test see at a given window size? | Simulation on your own embeddings: detection rate by share of changed texts and window size, with the A/A false alarm rate |
| `model_equality` / `model_equality_test` | Is the endpoint still serving the **same model** (not a quantized version, another model, a new system prompt)? | Answers to a fixed canary set, sampled several times; MMD with a Hamming kernel within prompts (Gao et al., ICLR 2025); permutation p-value within prompts. `model_equality_power` measures the detection rate by canary size on a pilot; `min_answers` turns a non-significant result on a small canary into "inconclusive" |
| `consistency` / `consistency_drift` | Does the system give the **same answer to the same question** asked several times (and in other words)? | Per-question pairwise agreement from `ConsistencyProbe` (pulsar-data-collection); window mean and share of questions in full agreement with bootstrap intervals over the questions; one-sided permutation test of a drop against a reference (paired on shared questions) |
| `ComplianceMonitor` | Does every run respect the **behavioural rules** of the agent? | Rules in linear temporal logic on finite traces (LTLf) or Declare templates (`precedence`, `response`, `absence`, `allowed_tools`...), versioned in YAML with a `severity` (info, minor, major, critical); per run, the violated rules (`violations`), the alert of the critical ones (`alerts`) and the fields to store with the run (`run_record`); per rule, the compliance rate with a Wilson interval, in absolute terms or against a reference period |

```python
from pulsar_metrics.genai import ComplianceMonitor, load_rules, trace_from

monitor = ComplianceMonitor(load_rules({"rules": [
    {"id": "auth-before-file", "template": "precedence", "events": ["authenticate", "create_file"], "severity": "critical"},
    {"id": "no-repeated-file", "formula": "G (create_file -> !X create_file)"},
]}), version="rules/1")
print(monitor.violations(trace_from("search_programs>create_file>authenticate")))
```

```
['auth-before-file']
```

LTLf operators: `!`, `&`, `|`, `->`, `X` (strong next), `F` (eventually), `G` (always), `U` (until); atoms are event
names (the tools called, in order, as in the `tool_sequence` field of the agent records), or atoms with predicates over a rich trace (below).

- **Per-run alerts**: `monitor.alerts(trace)` returns the violated rules of severity `critical` (or
  `min_severity=...`), most severe first, each with the rules version: one run that creates a file before
  authentication is an incident by itself. `monitor.run_record(trace)` gives `violated_rules`,
  `critical_violations`, `max_severity` and `rules_version`, to store with the `agent_run` record.
- **Against a reference**: `monitor.evaluate(traces, reference=reference_traces)` flags a rule only when it is
  violated more often than in the reference period (one-sided Fisher exact test, Benjamini-Hochberg over the rules,
  `p_value` adjusted, `threshold` = reference compliance). Rules of severity `absolute_from` (critical) or above
  keep the absolute verdict. Without `reference`, a rule violated at a stable rate alerts in every window.

```python
print([a["rule_id"] for a in monitor.alerts(trace_from("search_programs>create_file>authenticate"))])
```

```
['auth-before-file']
```

#### Rules over arguments, statuses and conversation events

A trace is a list of names (`trace_from`) or a **rich trace**: a list of mappings
`{"name": ..., "kind": "tool" | "event", "args": {...}, "status": "ok" | "error" | "timeout" | "soft_failure"}`, with
the conversation events (e.g. `user_confirmation`) in time order among the tool calls. `ComplianceMonitor` declares
`accepts_rich_trace = True`, so the trajectory collector of pulsar-data-collection passes it rich traces. A plain atom
matches an element by its name, on both forms; atom predicates read the rest of a rich element:

| Atom | Matches |
|---|---|
| `cancel_pending_order` | an element of that name (tool or event), as before |
| `cancel_pending_order[order_id=#W1]` | that tool with the argument equal to the value (values compared as text: `42` matches `42` and `"42"`; quote values with spaces or commas: `[reason='no longer needed']`) |
| `cancel_pending_order[order_id]` | that tool with the argument present |
| `cancel_pending_order[reason!=other, order_id]` | conditions joined by commas, all required; `!=` also matches when the call has arguments but not that one (a missing `reason` is not `other`) |
| `get_order_details[status=error]` | that tool with that status (`status` and `kind` are element fields; `args.status=...` reads an argument named status) |
| `@user_confirmation` | an **event** of that name (never a tool) |
| `*[status=error]`, `@*` | any tool with that status, any event |

Templates take a list of atoms for "any of them" (`events: ["@user_confirmation", [cancel_pending_order,
modify_pending_order_address]]`) and two new templates: `alternate_precedence` (each b preceded by an a since the
previous b: a confirmation before every write) and `alternate_response` (each a followed by a b before the next a).
`bind` restricts every tool atom to one value of the named arguments at a time, the rule holding for each value (events
are never bound: a plain atom still matches an event of its name). A rule with `@` atoms needs a collector that records
those events (`record_event`): a rich trace cannot tell "no confirmation happened" from "confirmations are not
recorded", and the rule then reports violations:

```yaml
rules:
  - id: status-checked            # an order is looked up before it is changed (the same order)
    template: precedence
    events: [get_order_details, [cancel_pending_order, modify_pending_order_address]]
    bind: [order_id]
    severity: major
  - id: confirm-before-write      # a user confirmation since the last write, before each write
    template: alternate_precedence
    events: ["@user_confirmation", [cancel_pending_order, modify_pending_order_address]]
    severity: critical
```

A rule sees the tool calls and only the events it names, so the rules over tool names give the same verdict on a rich
trace as on its tool sequence (an `agent_msg` event between two calls does not break `X` or `chain_response`).

**Not evaluable.** A rule that reads arguments, statuses or events (`Rule.needs_rich_trace`: a predicate, an `@`
atom or `bind`) cannot be judged on a name-only trace (`trace_from` returns a `NameTrace`; a non-empty list of
strings counts as one too), nor when the elements it depends on lack the data (a tool without `args`, a bound tool
without the bound argument; three-valued logic, so known elements still decide when they can). Such a rule is
reported, never counted as a violation or a pass: `Rule.verdict` returns None and `Rule.check` raises
`RuleNotEvaluable`; `monitor.not_evaluable(trace)` lists them; `violations` and `alerts` leave them out with a
`NotEvaluableWarning`; `run_record` adds `not_evaluable_rules` (only then); `evaluate` counts per rule the runs it
was judged on (`n`, the method says how many were not) and gives `status="not_evaluable"`, `metric_value` and
`drift_status` None to a rule judged on no run. On the synthetic replica of LA-02 (40 retail runs, 36 with a write),
the confirmation rule over tool names flags all 36; with `@user_confirmation` on the rich traces it flags the 23 real
violations, and on the name-only traces it is not evaluable on all 40.

#### Consistency of repeated answers

The input is the `questions` frame of `pulsar_data_collection.genai.ConsistencyProbe` (one row per question:
`agreement`, `full_agreement`, `n_answers`), or a mapping `{question_id: [normalized answers]}` (`question_agreement`).

```python
from pulsar_metrics.evidence import to_evidence
from pulsar_metrics.genai import consistency, consistency_drift

window = consistency(current_questions, model="llama3.2:3b", threshold=0.8)  # [agreement, full agreement share]
drop = consistency_drift(reference_questions, current_questions, model="llama3.2:3b", alpha=0.05)
records = to_evidence(window + [drop], "REG-0003", window={"id": "2026-W45"}, reference={"id": "2026-W40"})
```

| Result | metric_type / name | Value | Verdict (evidence) |
|---|---|---|---|
| mean agreement | `custom` / `consistency_agreement` | mean over the questions of the mean pair score, percentile bootstrap interval over the questions | with `threshold`: alert below it (`direction` lower, `decision_basis` value); else no verdict |
| full agreement | `custom` / `consistency_full_agreement` | share of questions whose answers all agree, idem | `full_agreement_threshold`, idem |
| drop | `drift` / `consistency_drift` (`statistic="full_agreement"`: `consistency_full_agreement_drift`) | current - reference (negative: less consistent), bootstrap interval | one-sided permutation p-value below `alpha` (`direction` lower, `decision_basis` p_value); paired sign flips on shared questions, else permutation of the questions |

All are exported in the `genai` family. Questions with fewer than 2 answers are set aside (counted in `method`). A
median below `min_repeats` answers per question (default 3) gives `status="few_repeats"`, fewer than
`min_questions` questions (default 10) `"few_questions"`: `reliable=False`, and the evidence reason code of the same
name; a window with no usable question is `not_evaluable`. The questions are the units of the bootstrap and of the
permutations: the answers to one question are not independent.

#### Known limits (night lab, TX-02, IM-02, LM-01, LI-03, LS-01 and LR-03)

| Limit | Lab figure | What the code does now |
|---|---|---|
| MMD misses a change of mix along one direction | polarity 50 % to 65 % positive: power 0.55 at 300 texts, 0.05 at 100; a 50 % change is needed for 80 % power at 300 | `component_test` and `detection_power`. On a synthetic replica tuned to the lab's MMD power, `max_t` reaches 0.45 to 0.65 against 0.5 for the MMD: **no unsupervised test on these embeddings fixes it**, the shift is near the statistical limit of a few hundred texts. Monitor the predicted class mix of the downstream model (`chi2` on its predictions) or a probe trained for the property; use `detection_power` to publish the minimal detectable change per window size |
| MMD misses a small share of foreign texts | 5 % of English texts among French reviews: power 0.40 at 300, 0.05 at 100 | `novelty_test`: on the synthetic replica, 0.8 at 300 and 0.5 at 100 (MMD: 0.2 and 0.15), and the foreign texts are named. Still to be confirmed on the lab's embeddings |
| The encoder and the PCA matter | power by PCA components (none / 8 / 32 / 128): 0.05 / 0.50 / 0.55 / 0.20; nomic-embed-text 0.25 (polarity) and 0.90 (5 % English) | `n_components` (32) and `bandwidth` are exposed; measure with `detection_power` before choosing |
| Cost of `mmd_test` at the default cap | 1000 x 1000 texts, 500 permutations: 11 to 18 s | vectorized permutations and eigendecomposition PCA: 0.7 s |
| Correlated rows of one unit break the permutation test | passages retrieved by a RAG agent (2 per run, top 2 of the same query) permuted as independent rows: 41 to 43 % A/A false alarms on windows of 100 runs (LR-03 replay, 2,000 draws) | `mmd_test(..., groups=(reference_groups, current_groups))` (also through `embedding_drift`) permutes and subsamples whole groups: 5.0 % [4.3, 5.8] on 3,000 draws. One row per run (its top passage) holds the level too. Without `groups`, results are unchanged |
| MMD on pixels needs large image windows | 3 corruptions of 15 need more than 50 images for 80 % power; 7 of 15 missed at 20 images | `detection_power` sizes the window on your own images; an image encoder (embeddings instead of pixels) is the stronger lever and is left to the user |
| A change of quantization is missed by the default canary | llama3.2 3B q4_K_M vs q8_0: detected 20 % at 12 prompts x 5 answers, 40 % at 24 x 5, 60 % at 48 x 5, 100 % at 48 x 10 (5 runs per cell; the 480 figure has a 95 % lower bound near 0.5) | `RECOMMENDED_CANARY` (48 x 10), `model_equality_power` on a pilot, `min_answers=480` for an explicit "inconclusive"; the default canary size of the probe lives in pulsar-data-collection |
| Rules violated at a stable rate alert in every window | auth-before-file 9 windows of 10 (6.0 % against 7.2 % in the reference), amount-after-eligibility 10 of 10 (29.7 % against 28.9 %) | `evaluate(reference=...)` |

## Explanations and their drift

`pulsar_metrics.explain` watches **what the model relies on**, from feature attributions (TreeSHAP with the `explain`
extra: `pip install pulsar-metrics[explain]`, or any attribution matrix with one row per prediction and one column per
feature). Explanations are estimates too, so every comparison carries a bootstrap interval, and the instrument is
checked first.

| Tool | Answers | Verdict |
|---|---|---|
| `AttributionDrift` | Has the ranking of features changed (NDCG@k, overlap of the top k)? Has the share of importance of a feature moved by more than `share_tolerance`? | from the interval; per-feature intervals simultaneous (Bonferroni) |
| `importance_by_group` | Does a feature weigh more for one group than for the reference group (possible proxy of a protected attribute)? | from the interval; small cells suppressed |
| `explain_decision`, `describe` | Plain-language sentences, French or English, for one decision or for the results above | — |

- **Instrument check**: `fit(reference)` bootstraps the reference window; when the ranking of features is not stable
  at that sample size, the ranking verdicts are `None` and `method` says why.
- **Watch the shares, not only the ranking**: a feature can gain a lot of weight while staying in the top k. On
  ACSIncome, a model retrained on labels biased against women raised the share of `SEX` from 7.6 % to 18.2 % (change
  established) while NDCG@5 stayed at 0.94.
- The wording describes how the model works, never a cause, and quotes the interval with every figure.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from pulsar_metrics.explain import AttributionDrift, describe, explain_decision, tree_attributions

rng = np.random.default_rng(0)
n = 4000
X = pd.DataFrame(
    {"income": rng.normal(50, 15, n), "hours": rng.normal(38, 8, n), "age": rng.integers(18, 70, n), "region": rng.integers(0, 2, n)}
)
y = (0.08 * X["income"] + 0.1 * X["hours"] + rng.normal(0, 1, n) > 8).astype(int)
model_v1 = GradientBoostingClassifier(random_state=0).fit(X, y)
# retrained on labels where half of the approvals of region 0 were lost: the new model relies on the region
y_v2 = y * ((X["region"] == 1) | (rng.random(n) < 0.5))
model_v2 = GradientBoostingClassifier(random_state=0).fit(X, y_v2)

drift = AttributionDrift(k=3).fit(tree_attributions(model_v1, X[:2000]))
results = drift.run(tree_attributions(model_v2, X[2000:]))
labels = {"income": {"fr": "revenu", "en": "income"}, "hours": {"fr": "heures par semaine", "en": "hours per week"}}
print("\n".join(describe(results, lang="fr", labels=labels, only_changes=True)))
print(explain_decision(tree_attributions(model_v1, X[:1]).iloc[0], X.iloc[0], lang="en", labels=labels))
```

```
Recouvrement des 3 premiers facteurs : 0,50 [0,50 ; 0,50], seuil 0,60. Changement établi.
Poids de « revenu » : −12,6 points de pourcentage [−14,3 ; −11,2]. Changement établi.
Poids de « heures par semaine » : −8,9 points de pourcentage [−10,3 ; −7,3]. Changement établi.
Poids de « region » : +23,2 points de pourcentage [+22,6 ; +23,6]. Changement établi.
Factors that weighed most in this decision:
1. hours per week (value: 44.82) weighed in favour (+1.22)
2. income (value: 51.89) weighed in favour (+0.36)
3. age (value: 35) weighed against (-0.18)
These factors describe how the model works, not a cause nor a judgement about the person.
```

## Custom metrics

`CustomDriftMetric` turns a function of `(current_column, reference_column, **kwargs)` into a drift metric;
`CustomMetric` turns a function of keyword arguments into a metric. Both follow the verdict convention above.

```python
from pulsar_metrics.metrics.base import CustomMetric
from pulsar_metrics.metrics.drift import CustomDriftMetric


@CustomDriftMetric
def mean_shift(current, reference, **kwargs):
    return abs(current.mean() - reference.mean())


shift = mean_shift(metric_name="mean_shift", feature_name="income")
print(shift.evaluate(current=current, reference=reference, threshold=0.5).drift_status)


@CustomMetric
def positive_rate(current, **kwargs):
    return current["y_pred"].mean()


rate = positive_rate(metric_name="positive_rate")
print(rate.evaluate(current=current, threshold=[0.3, 0.7]).drift_status)  # drift outside the interval
```

Custom metrics are evaluated directly; `Analyzer` handles the built-in metrics.

## Development

The project uses [uv](https://docs.astral.sh/uv/) and a standard PEP 621 `pyproject.toml`.

```bash
uv sync                                         # .venv with runtime and dev dependencies
uv run pytest                                   # unit and characterization tests, with coverage
uv run ruff check . && uv run black --check .   # lint and formatting
```

`tests/test_characterization.py` freezes the output of every metric on a fixed synthetic dataset in
`tests/golden/metrics_reference.json` (relative tolerance 1e-9). Any numerical change after a dependency upgrade fails
the test. Regenerate the file only after an intended behaviour change:

```bash
uv run python -m tests.test_characterization
```

Continuous integration runs lint and the tests on Python 3.12 to 3.15 with pandas 3, plus Python 3.12 with pandas 2,
for every pull request; tests are required to pass. Pushing a tag builds and publishes the package.

## About [PulsarML](https://pulsar.ml/)

PulsarML helps monitor machine learning models in production and gain insights into their performance. It is made of
two open-source packages:

- [pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection): lightweight SDK that
  collects features, predictions and metadata from a model serving code into InfluxDB.
- [pulsar-metrics](https://github.com/Rocket-Science-Development/pulsar_metrics): data drift, concept drift and
  performance metrics, with custom metrics.

[pulsar-demo](https://github.com/Rocket-Science-Development/pulsar_demo) shows an example use case combining both
packages.

Powered by [Rocket Science Development](https://rocketscience.one/).

## Contributing

1. Open an issue describing the change.
2. Fork the repository, develop on a branch and add tests.
3. Check `uv run pytest` and the linters, then submit a pull request referencing the issue.

Licensed under the [Apache License 2.0](LICENSE).
