# pulsar-metrics

Author: abenlagra@rocketscience.one

Pulsar-metrics is an open-source Python library for evaluating and monitoring data and concept drift with an extensive set of metrics. It also offers the possibility to use custom metrics defined by the user.

## Getting started

### Pulsar-metrics components

There are two core components in pulsar-metrics: metrics and analyzers

#### Metrics
An API to calculate single metrics for data and concept drift. Metrics results are unified in a single data structure `MetricsResults` storing not only the metrics value, but also a couple of metadata related to the model and the data used for its calculation

```
MetricResults(metric_name=None, type='performance', model_id='model_1', model_version='1', data_id=None, feature=None, value=None, status=None, threshold=None, period_start=None, period_end=datetime.datetime(2022, 7, 1, 0, 0), eval_timestamp=datetime.datetime(2022, 9, 26, 10, 28, 27, 846122), conf_int=None)
```

There are three types of metrics:

##### Data drift metrics for the calculation of ditributional changes of the features used in the model. The metrics included so far are:
###### Kullback-Leibler (KL) divergence : This statistics measures how different is a probability distribution $P$ with respecvt to the reference probability distributiuon $Q$
* Wasserstein distance
* T-test for location drift
* Mann-Whitney U test
* Levene test for dispersion drift
* Kolmogorov-Smirnov test
* Cramer von Mises test
* Chi-square test for categorical features

Data drift metrics are implemented either in the `DriftMetric` (For the KL divergence and the Wasserstein distance) or `DriftTestMetric` classes.

- ***Performance metrics*** for te calculation of the performance of classification and regression models. In particular, the following metrics are implemented:

* Accuracy
* Precision
* Recall
* f1-score
* Log loss
* AUC
* AUCPR
* Brier Score
* Mean squarred error (MSE)
* Mean absolute error (MAE)
* Mean absolute percentage error (MAPE)
* R-square score

Performance metrics are implemented in the `PerformanceMetric` class.

- ***Custom metrics***. The user has the ability to define his own metric through the `@CustomMetric` decorator (see below for an example)

All three types of metrics inherit the `AbstractMetrics` class.
#### Analyzers

An analyzer groups multiple metrics calculations in a single run. It allows to use which metrics to use and for which features.

### Example usage

To use the library, you need a reference dataset, typically the training dataset, and an analysis dataset which we want to compare with former.


#### Calculating a single metric
For a single metric, we first start by instantiating the appropriate metrics class by specifying the name of the metric ("ttest" in the example below)

```python
from pulsar_metrics.metrics.drift import DriftTestMetric
driftTest = DriftTestMetric(name = 'ttest', data = data_new, feature_name = feature_name)
```
Then we run the `.evaluate()` method to calculate the metric

```python
driftTest.evaluate(alpha = 0.05, reference = data_ref[feature_name])
```

The result is returned through the `.get_result()` method of te metric object

```python
driftTest.get_result()
```

#### Using the analyzer
When multiple metrics are required for different features, the analyzer allows one to calculate all the metrics at once.

First, instantiate an analyzer object

```python
from pulsar_metrics.analyzers.base import Analyzer
analysis = Analyzer(name = 'First Analyzer', description='My first Analyzer', data = data_new)
```

Then add the metrics of interest

```python
analysis.add_drift_metrics(metrics_list=['wasserstein', 'ttest', 'ks_2samp'], features_list=['Population', 'MedInc']);
analysis.add_performance_metrics(metrics_list=['accuracy'], y_name = 'clf_target');
```

Then, you can run the analyzer while optionnally specfyings options for each metrics as a dictionnary for the `options` keywords

```python
analysis.run(data_ref = data_ref, options = {'ttest': {'alpha': 0.01, 'equal_var': False}})
```

It then possible to get the results of the analysis as a pandas dataFrame

```python
analysis.results_to_pandas()
```

![image](https://user-images.githubusercontent.com/105239615/206878435-b3bd2b8d-5196-45cd-9eb6-76d70e002c23.png)

#### Creating a custom metric
The `@CustomMetric` decorator allows to transform any function to the `AbstractMetrics` class

```python
from pulsar_metrics.metrics.base import CustomMetric
@CustomMetric
def test_custom(a, b, **kwargs):
    return np.max(a - b)
```

