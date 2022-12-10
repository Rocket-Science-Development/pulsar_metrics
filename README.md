# pulsar-metrics

Authors: abenlagra@rocketscience.one

Pulsar-metrics is an open-source Python library for evaluating and monitoring data and concept drift with an extensive set of metrics. It also offers the possibility to use custom metrics defined by the user.

## Getting started

### Pulsar-metrics components

There are two core components in pulsar-metrics: metrics and analyzers

#### Metrics
An API to calculate single metrics for data and concept drift. Metrics results are unified in a single data structure `MetricsResults` storing not only the metrics value, but also a couple of metadata related to the model and the data used for its calculation

```
MetricResults(metric_name=None, type='performance', model_id='model_1', model_version='1', data_id=None, feature=None, value=None, status=None, threshold=None, period_start=None, period_end=datetime.datetime(2022, 7, 1, 0, 0), eval_timestamp=datetime.datetime(2022, 9, 26, 10, 28, 27, 846122), conf_int=None)
```
#### Analyzers
