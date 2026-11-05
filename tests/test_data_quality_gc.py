"""Data-quality checks on real open data: hourly weather of Montréal-Trudeau airport (ECCC, MONTREAL INTL A, climate
ID 7025251), the input data of the NO2 trial of pulsar_demo/essai_donnees_gc. Only aggregated measurements: no person.

The raw monthly files (yul_YYYY_MM.csv, written by pulsar_demo/essai_donnees_gc/download.py) are not in git: the test
reads them from $PULSAR_GC_DATA, or from a data/essai_gc folder above the repository, and is skipped without them.
Reference: 2016-2018. Every month of 2019-2021 is checked as it came; July 2019 is then broken the way feeds break
(a unit change, a sensor outage, relabelled categories, a batch sent twice), each check catching its own break.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from pulsar_metrics.evidence import to_evidence
from pulsar_metrics.quality import DataQualityAnalyzer, DataQualityProfile, check

COLUMNS = {
    "Temp (°C)": "temp",
    "Dew Point Temp (°C)": "dew",
    "Rel Hum (%)": "rh",
    "Wind Dir (10s deg)": "wdir",
    "Wind Spd (km/h)": "wspd",
    "Stn Press (kPa)": "press",
    "Weather": "weather",
}
RULES = ["dew <= temp", "rh <= 100"]
# Tuned on the reference only: "weather" is empty when no phenomenon is reported (52 % of the hours, more in summer),
# and its levels are combinations ("Snow,Blowing Snow") of which a few new ones appear every winter
POLICY = {"columns": {"weather": {"null_share": {"margin": 0.25}, "unseen_category_share": 0.03}}}


def data_folder():
    if os.environ.get("PULSAR_GC_DATA"):
        return Path(os.environ["PULSAR_GC_DATA"])
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "essai_gc" / "yul_2016_01.csv").is_file():
            return parent / "data" / "essai_gc"
    return None


DATA = data_folder()
pytestmark = pytest.mark.skipif(
    DATA is None or not (DATA / "yul_2021_12.csv").is_file(), reason="ECCC weather files not found (see download.py)"
)


def month(year, number):
    raw = pd.read_csv(DATA / f"yul_{year}_{number:02d}.csv", encoding="utf-8-sig")
    out = raw[list(COLUMNS)].rename(columns=COLUMNS)
    out["pred_timestamp"] = pd.to_datetime(raw["Date/Time (LST)"])
    return out


@pytest.fixture(scope="module")
def profile():
    reference = pd.concat([month(y, m) for y in (2016, 2017, 2018) for m in range(1, 13)], ignore_index=True)
    return DataQualityProfile.fit(reference, rules=RULES, max_categories=100, reference_id="yul-2016-2018")


def alerts(results):
    return {(r.metric_name, r.feature_name) for r in results if r.drift_status}


def test_reference_profile_of_the_airport_weather(profile):
    assert profile.n_rows == 26304  # 3 years of hours
    assert profile.columns["press"].kind == "numeric" and 95 < profile.columns["press"].min < profile.columns["press"].max < 105
    assert profile.columns["weather"].kind == "categorical" and len(profile.columns["weather"].categories) > 50
    assert [r.reference_share for r in profile.rules] == [0.0, 0.0]


def test_the_36_months_of_2019_to_2021_pass_as_they_came(profile):
    false_alarms = {}
    for year in (2019, 2020, 2021):
        for number in range(1, 13):
            found = alerts(check(profile, month(year, number), POLICY))
            if found:
                false_alarms[f"{year}-{number:02d}"] = found
    assert false_alarms == {}


@pytest.mark.parametrize(
    "damage, expected",
    [
        pytest.param(lambda d: d.assign(press=d["press"] * 10), {("out_of_range_share", "press")}, id="pressure in hPa"),
        pytest.param(
            lambda d: d.assign(rh=d["rh"].mask(d.index < len(d) // 3)), {("null_share", "rh")}, id="humidity sensor outage"
        ),
        pytest.param(
            lambda d: d.assign(weather=d["weather"].replace({"Rain": "Pluie", "Mainly Clear": "Généralement dégagé"})),
            {("unseen_category_share", "weather")},
            id="weather labels in French",
        ),
        pytest.param(
            lambda d: pd.concat([d, d.iloc[:24]], ignore_index=True), {("duplicate_share", None)}, id="a day sent twice"
        ),
        pytest.param(
            lambda d: d.assign(temp=d["dew"], dew=d["temp"]),
            {("rule_violation_share", "dew <= temp"), ("out_of_range_share", "dew")},  # July temperatures above any dew point
            id="dew point and temperature swapped",
        ),
        pytest.param(
            lambda d: d.assign(rh=d["rh"].astype("Int64").astype(str) + " %"), {("schema", "rh")}, id="humidity as text"
        ),
    ],
)
def test_each_break_of_july_2019_is_caught_by_its_check(profile, damage, expected):
    clean = month(2019, 7)
    assert alerts(check(profile, clean, POLICY)) == set()
    assert alerts(check(profile, damage(clean), POLICY)) == expected


def test_checks_a_type_change_leaves_without_value_say_so(profile):
    clean = month(2019, 7)
    results = {(r.metric_name, r.feature_name): r for r in check(profile, clean.assign(rh=clean["rh"].astype(str) + " %"))}
    assert results[("out_of_range_share", "rh")].status == "type_mismatch"
    assert results[("rule_violation_share", "rh <= 100")].status == "no_values_in_window"


def test_evidence_of_a_broken_week_for_the_governance_service(profile):
    analyzer = DataQualityAnalyzer("airport weather quality", model_id=None, profile=profile, policy=POLICY)
    week = month(2019, 7).iloc[: 7 * 24]
    analyzer.run(week.assign(press=week["press"] * 10))
    records = to_evidence(analyzer, "NO2-MTL")
    alert = [r for r in records if r.metric.status == "alert"]
    assert [(r.metric.name, r.metric.feature) for r in alert] == [("out_of_range_share", "press")]
    assert alert[0].metric.value == 1.0 and alert[0].metric.direction == "upper" and alert[0].reliability.reliable
    assert alert[0].reference.id == "yul-2016-2018" and alert[0].window.n_rows == 168
