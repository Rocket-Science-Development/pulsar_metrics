"""Bias-injection bench for FairnessMonitor: false alarms under the null, power and detection delay.

A stream of decisions is fed by batches. After a warm-up, the selection rate of one group is lowered by delta. The
bench reports, per (delta, share of the group):
- the alarm rate over the whole horizon (delta <= tolerance: false alarms, to compare with alpha),
- the power (share of runs with an alarm after the injection) and the median delay, in decisions after the injection.

It compares the anytime-valid monitor with what is usually done: a fixed-sample test (two-proportion z-test against the
tolerance) re-run on the cumulated data after every batch.

Two sources:
- synthetic: two groups, selection rate 0.4;
- folktables: decisions of a logistic regression on ACSIncome (Massachusetts 2018), rows drawn with replacement from
  the test split; groups RAC1P=6 (monitored) vs RAC1P=1 (reference), whose observed gap (about 0.04) is within the
  tolerance. Needs `folktables` and `scikit-learn`, and the ACS files in --data-dir (never in a repository).

    python benchmarks/fairness_drift.py --runs 100 --out bench.json
    python benchmarks/fairness_drift.py --source folktables --data-dir C:/dev/pulsar-ml/data/folktables
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from pulsar_metrics.fairness import FairnessMonitor

Z = 1.959964  # two-sided 95 %


def naive_alarm(sum_g, n_g, sum_r, n_r, tolerance):
    """Fixed-sample test re-run at each look: reject when |difference| - tolerance exceeds 1.96 standard errors"""
    if n_g < 30 or n_r < 30:
        return False
    p_g, p_r = sum_g / n_g, sum_r / n_r
    se = math.sqrt(p_g * (1 - p_g) / n_g + p_r * (1 - p_r) / n_r) or 1e-12
    return (abs(p_g - p_r) - tolerance) / se > Z


def run_once(sample, delta, share, rng, warmup, horizon, batch, tolerance, alpha):
    """Decision index of the first alarm of the monitor and of the naive test (None: no alarm)"""
    monitor = FairnessMonitor("group", "R", ["G"], alpha=alpha, tolerance=tolerance)
    sums = {"G": 0, "R": 0}
    counts = {"G": 0, "R": 0}
    first = {"monitor": None, "naive": None}
    seen = 0
    while seen < warmup + horizon:
        in_group = rng.random(batch) < share
        y = sample(in_group, rng)
        if seen >= warmup and delta > 0:
            # injection: approvals of the monitored group are withdrawn with the probability that lowers its rate by delta
            withdraw = in_group & (y == 1) & (rng.random(batch) < delta / sample.rate_g)
            y = np.where(withdraw, 0, y)
        seen += batch
        frame = pd.DataFrame({"group": np.where(in_group, "G", "R"), "y_pred": y})
        if first["monitor"] is None and monitor.update(frame)[0].drift_status:
            first["monitor"] = seen
        for key, mask in (("G", in_group), ("R", ~in_group)):
            sums[key] += int(y[mask].sum())
            counts[key] += int(mask.sum())
        if first["naive"] is None and naive_alarm(sums["G"], counts["G"], sums["R"], counts["R"], tolerance):
            first["naive"] = seen
        if first["monitor"] is not None and first["naive"] is not None:
            break
    return first


class Synthetic:
    rate_g = 0.4

    def __call__(self, in_group, rng):
        return (rng.random(len(in_group)) < 0.4).astype(int)


class Folktables:
    def __init__(self, data_dir):
        from folktables import ACSDataSource, ACSIncome
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=data_dir)
        X, y, _ = ACSIncome.df_to_pandas(source.get_data(states=["MA"], download=False))
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train.values.ravel())
        decisions = model.predict(X_test).astype(int)
        race = X_test["RAC1P"].to_numpy()
        self.pool_g = decisions[race == 6]
        self.pool_r = decisions[race == 1]
        self.rate_g = float(self.pool_g.mean())
        self.observed_gap = float(self.pool_g.mean() - self.pool_r.mean())

    def __call__(self, in_group, rng):
        out = rng.choice(self.pool_r, len(in_group))
        out[in_group] = rng.choice(self.pool_g, int(in_group.sum()))
        return out


def summarise(firsts, warmup):
    out = {}
    for method in ("monitor", "naive"):
        times = [f[method] for f in firsts]
        after = [t - warmup for t in times if t is not None and t > warmup]
        out[method] = {
            "alarm_rate": sum(t is not None for t in times) / len(times),
            "alarm_before_injection": sum(t is not None and t <= warmup for t in times) / len(times),
            "power": len(after) / len(times),
            "median_delay": float(np.median(after)) if after else None,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["synthetic", "folktables"], default="synthetic")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.15, 0.2])
    parser.add_argument("--shares", type=float, nargs="+", default=[0.05, 0.2, 0.5])
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sample = Folktables(args.data_dir) if args.source == "folktables" else Synthetic()
    report = {"settings": vars(args), "cells": []}
    if args.source == "folktables":
        report["observed_gap"] = sample.observed_gap
        report["rate_monitored_group"] = sample.rate_g
        print(f"observed gap {sample.observed_gap:.3f}, rate of the monitored group {sample.rate_g:.3f}", flush=True)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    for share in args.shares:
        for delta in args.deltas:
            firsts = [
                run_once(sample, delta, share, rng, args.warmup, args.horizon, args.batch, args.tolerance, args.alpha)
                for _ in range(args.runs)
            ]
            cell = {"share": share, "delta": delta, **summarise(firsts, args.warmup)}
            report["cells"].append(cell)
            m, n = cell["monitor"], cell["naive"]
            print(
                f"share {share:.2f} delta {delta:.2f} | monitor: alarms {m['alarm_rate']:.2f} "
                f"(before {m['alarm_before_injection']:.2f}) power {m['power']:.2f} delay {m['median_delay']} | "
                f"naive: alarms {n['alarm_rate']:.2f} (before {n['alarm_before_injection']:.2f}) "
                f"power {n['power']:.2f} delay {n['median_delay']}",
                flush=True,
            )
    report["seconds"] = round(time.perf_counter() - started, 1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
