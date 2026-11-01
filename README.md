# pulsar-metrics

[![Continuous Integration](https://github.com/Rocket-Science-Development/pulsar_metrics/actions/workflows/continuous-integration.yaml/badge.svg)](https://github.com/Rocket-Science-Development/pulsar_metrics/actions/workflows/continuous-integration.yaml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**pulsar-metrics** is an open-source Python library for evaluating and monitoring **data drift**, **concept drift** and
**model performance**. Every metric returns the same result structure with a value, an explicit drift verdict and,
when something goes wrong, an error message instead of a silent failure. User-defined metrics are supported.

It pairs with [pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection), which
collects features and predictions from a model serving code into InfluxDB.

- [Installation](#installation)
- [Quick start](#quick-start)
- [Results and drift verdicts](#results-and-drift-verdicts)
- [Available metrics](#available-metrics)
- [Computing metrics one by one](#computing-metrics-one-by-one)
- [The Analyzer](#the-analyzer)
- [Confirming drift over successive windows](#confirming-drift-over-successive-windows)
- [Custom metrics](#custom-metrics)
- [Development](#development)
- [Changelog](CHANGELOG.md)

## Installation

Requires **Python 3.12 or later** (tested on 3.12 and 3.13).

```bash
pip install "pulsar-metrics @ git+https://github.com/Rocket-Science-Development/pulsar_metrics.git"
```

Pin a commit or a tag for reproducible installs (`...pulsar_metrics.git@<sha>`). The PyPI release (0.1.3) predates the
current API and only supports Python 3.9.

Main dependencies: pandas >= 2.2 (< 3), numpy, scipy, scikit-learn, pydantic 2.

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
| `metric_type` | `drift`, `performance`, `statistics` or `custom` |
| `metric_name` | e.g. `ks_2samp`, `psi`, `accuracy` |
| `feature_name` | Feature analyzed; `prediction` for performance metrics |
| `metric_value` | Value of the metric (p-value for statistical tests) |
| `threshold` | Threshold, `alpha` or `[min, max]` interval used for the verdict |
| `drift_status` | **`True` = drift or degradation detected, `False` = none, `None` = no verdict** |
| `conf_int` | Bootstrap confidence interval (performance metrics with `bootstrap=True`) |
| `error` | Message when the metric could not be computed (`metric_value` and `drift_status` are then `None`) |

The verdict follows **one convention for every metric**:

- **Statistical tests** (`DriftTestMetric`): drift when the p-value is below `alpha` (default 0.05).
- **Distance metrics** (`DriftMetric`) and custom metrics: drift when the value **exceeds** the threshold
  (`upper_bound=True`, default), falls below it (`upper_bound=False`), or lies **outside** a `[min, max]` interval.
- **Performance metrics**: the threshold is a **minimum** for accuracy, precision, recall, f1, auc, aucpr and r2, and a
  **maximum** for error metrics (log_loss, brier, mse, mae, mape). Pass `upper_bound` to override.
- **No threshold**: the value is computed and `drift_status` is `None`.

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
bin absent from one sample gives a large but finite value instead of `inf`.

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

The statistical tests other than `chi2` apply to numeric features. For a categorical feature use `chi2`, `psi` or `kl`.

### Performance (`PerformanceMetric`)

| Name | Metric | Threshold | Predictions expected |
|---|---|---|---|
| `accuracy`, `precision`, `recall`, `f1` | Classification | minimum | labels (`y_pred`) |
| `auc`, `aucpr` | ROC AUC, average precision | minimum | scores or probabilities |
| `log_loss`, `brier` | Probabilistic losses | maximum | probabilities |
| `mse`, `mae`, `mape` | Regression errors (`mape` = mean absolute percentage error) | maximum | values |
| `r2` | Coefficient of determination | minimum | values |

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
| `model_id`, `model_version` | Rows are filtered on the analyzer's `model_id` and `model_version` |
| `pred_timestamp` | Prediction time (dates or ISO strings, mixed precisions accepted); gives `period_start` / `period_end` |
| features | Columns named in `features_list` |
| `y_true`, `y_pred` (or names given with `y_name` / `pred_name`) | Performance metrics. Rows without ground truth make the metric fail with an explicit error |

**Methods**

| Method | Description |
|---|---|
| `add_drift_metrics(metrics_list, features_list)` | Adds every metric for every feature; unknown names raise an error |
| `add_performance_metrics(metrics_list, **kwargs)` | Adds performance metrics; `y_name` / `pred_name` select the columns |
| `run(current, reference, options={})` | Computes summary statistics and all metrics. `options` maps a **metric name** to its keyword arguments (`threshold`, `alpha`, `upper_bound`, `bootstrap`...); they apply to every feature for that metric |
| `get_result()` | List of `MetricResults` |
| `results_to_pandas()` | DataFrame of results plus model metadata (`model_id`, `model_version`, `period_start`, `period_end`, `eval_timestamp`) |
| `results_to_json()` | List of JSON strings |

Because `options` are set per metric name, distance thresholds are shared by all features: standardize numeric
features against the reference (`(x - mean_ref) / std_ref`) or run one analyzer per group of features.

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

Continuous integration runs lint and the tests on Python 3.12 and 3.13 for every pull request; tests are required to
pass. Pushing a tag builds and publishes the package.

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
