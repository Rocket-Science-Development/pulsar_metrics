# pulsar-metrics

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

##### - Data drift metrics for the calculation of ditributional changes of the features used in the model. The metrics included so far are:
###### [Kullback-Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence): This statistics measures how different is a probability distribution $P$ with respect to a reference probability distributiuon $Q$ (typically the probability distribution of the treaining features). More precisely, the KL divergence $D_{KL}(P||Q)$ is given by the fllowing formula $$D_{KL}(P||Q) = \sum_x P(x) \log \left ( \frac{P(x)}{Q(x)} \right )$$ $D_{KL}(P||Q)$ is always non-negative et is zero when the distributions are identical. Hence, a drift would be detected if its value is larger than a given threshold decided by the use

###### [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric) is a distance measure between two probability measures $Q$ and $P$. More precisely, the (first) Wassersetin distance $W_1(P, Q)$ is given by the formula $$W_1(P, Q) = \int_{-\infty}^{+\infty}|F_Q(x) - F_P(x)|dx$$ where $F_Q$ is the cumulative distribution function of $Q$. The metric is strctly non negative and a drift would be detected if its value is larger than a given threshold decided by the user.

###### [T-test](https://en.wikipedia.org/wiki/Student%27s_t-test) is a 2 samples paremetric statistical test to detect a difference in the means of the distributions of the two samples. More precisely, the test used is the [Welch test](https://en.wikipedia.org/wiki/Welch%27s_t-test) in which the 2 samples do not necessarily have the same variance or size. Since it is a statistical test, a location drift is detected when the p-value is smaller than a significance level chosen by the user (default is 0.05).

###### [Mann-Whitney U test](https://en.wikipedia.org/wiki/Mann–Whitney_U_test) for location shift is a 2 sample non-parametric statistical test to detect a difference in the medians of the distrbutions of two samples. Since it is a statistical test, a location drift is detected when the p-value is smaller than a significance level chosen by the user (default is 0.05).

###### [Levene's test](https://en.wikipedia.org/wiki/Levene%27s_test) is a 2 samples parametric statistical test to detect a difference in the variances of the distributions of the two samples. Since it is a statistical test, a dispersion drift is detected when the p-value is smaller than a significance level chosen by the user (default is 0.05).

###### [Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test) is a 2 samples nonparametric statistical test to check whether two samples come from the same distribution. The test statistics is given by $$D_{n, m} = \sup_x |F_{1, n}(x) - F_{2, m}(x)|$$ where $F_{1, m}$ is the empirical cumulative distrbutin functin of sample 1 with size $n$. Since it is a statistical test, a dispersion drift is detected when the p-value is smaller than a significance level chosen by the user (default is 0.05).

###### [Cramer von Mises test](https://en.wikipedia.org/wiki/Cramér–von_Mises_criterion) is a 2 samples nonparametric statistical test to check whether two samples come from the same distribution. The test statistics is given by $$T_{n, m} = \frac{nm}{n+m} \int_{-\infty}^{+\infty} |F_{1, n}(x) - F_{2, m}(x)|^2 dF_{n+m}$$ where $F_{1, m}$ is the empirical cumulative distrbutin functin of sample 1 with size $n$ and $F_{n+m}$ is the emprirical distribution function of the two samples together. Since it is a statistical test, a distributios drift is detected when the p-value is smaller than a significance level chosen by the user (default is 0.05).

###### [Chi-square test](https://en.wikipedia.org/wiki/Chi-squared_test) to compare the distribution of a categorical feature in 2 samples by comparing the frequencies of unique modalities. Since it is a statistical test, a distribution drift is detected when the p-value is smaller than a significance level chosen by the user (default is 0.05).

Data drift metrics are implemented either in the `DriftMetric` (For the KL divergence and the Wasserstein distance) or `DriftTestMetric` classes. The choice of the metric is specified with the `name` parameter in the init method according to the following table

|Metric|Name|
|------|----|
|Kullback-Leibler divergence|'kl'|
|Wasserstein distance|'wasserstein'|
|T-test|'ttest'|
|Mann Whitney U test|'manwu'|
|Leven's test|'levene'|
|Kolmgorv-Smirnov|'ks_2samp'|
|Cramer von Mises test|'CvM'|
|Chi square test|'chi2'|


##### - Performance metrics for te calculation of the performance of classification and regression models. In particular, the following metrics are implemented:

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

##### - Custom metrics. The user has the ability to define his own metric through the `@CustomMetric` decorator (see below for an example)

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

## About [PulsarML](https://pulsar.ml/)

PulsarML is a project helping with monitoring your models and gain powerful insights into its performance.

We released two Open Source packages :
- [pulsar-data-collection](https://github.com/Rocket-Science-Development/pulsar_data_collection) :  lightweight python SDK enabling data collection of features, predictions and metadata from an ML model serving code/micro-service
- [pulsar-metrics](https://github.com/Rocket-Science-Development/pulsar_metrics) : library for evaluating and monitoring data and concept drift with an extensive set of metrics. It also offers the possibility to use custom metrics defined by the user.

We also created [pulsar demo](https://github.com/Rocket-Science-Development/pulsar_demo) to display an example use-case showing how to leverage both packages to implement model monitoring and performance management.

Want to interact with the community? join our [slack channel](https://pulsarml.slack.com)

Powered by [Rocket Science Development](https://rocketscience.one/)

## Contributing

1. Fork this repository, develop, and test your changes
2. open an issue
3. Submit a pull request with a reference to the issue
