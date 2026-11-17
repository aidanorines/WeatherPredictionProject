"""Clean the raw GHCN-Daily station files into one table for modeling.

In a notebook:
    from prepare_data import load_data
    data = load_data()              # builds the table the first time, then loads the saved copy
    data = load_data(rebuild=True)  # rebuild after changing this file

Or from a terminal, to build or rebuild the table:
    python prepare_data.py

The table has one row per station and calendar day, with no days missing, so
shifting a column by k rows always moves k days. Rows marked `usable` are the
forecast days every method is scored on.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"
STATIONS = DATA / "stations.csv"  # every station download_data.py fetches, written before it downloads
TABLE = DATA / "processed" / "stations.pkl"

TENTHS = ["TMAX", "TMIN", "PRCP", "AWND"]  # stored as value x 10
PLAIN = ["SNOW", "SNWD"]  # stored in whole mm
ELEMENTS = TENTHS + PLAIN
PERIODS = {"train": ("2000-01-01", "2019-12-31"),
           "validation": ("2020-01-01", "2022-12-31"),
           "test": ("2023-01-01", "2025-12-31")}
CALENDAR = pd.date_range("2000-01-01", "2025-12-31", name="DATE")
MIN_COVERAGE = 0.8  # share of days with a TMAX, needed in train and in validation
TEXAS_RECORDS = (-30.6, 48.9)  # °C, Texas's record low and record high
# Sources whose day ends at the morning reading. Airport sources (W, X, ...) end at midnight,
# even when a reading time is filed (Stephenville Clark Fld files 0800).
COOP_SOURCES = {"0", "7", "H", "K", "Z"}
# Station-years whose morning highs are already filed on the right day, written by
# check_date_alignment.py from comparisons with nearby midnight airports.
NO_SHIFT_FILE = Path(__file__).parent / "no_shift.csv"
LEADS = range(1, 8)
TARGETS = [f"y{h}" for h in LEADS]


def clean_station(path, no_shift_years=()):
    """One station's file as a daily table: passing values only, in real units."""
    rows = pd.read_csv(path, usecols=["DATE", "ELEMENT", "VALUE", "QFLAG", "SFLAG", "OBS_TIME"],
                       dtype={"DATE": str, "VALUE": "float32", "SFLAG": str, "OBS_TIME": str}, low_memory=False)
    rows = rows[rows.QFLAG.isna() & rows.ELEMENT.isin(ELEMENTS)]  # blank QFLAG = passed all checks
    date = pd.to_datetime(rows.DATE, format="%Y%m%d")
    # Morning observers report a 24-hour maximum that usually happened the afternoon before.
    morning = ((rows.ELEMENT == "TMAX") & (pd.to_numeric(rows.OBS_TIME, errors="coerce") < 1200)
               & rows.SFLAG.isin(COOP_SOURCES) & ~date.dt.year.isin(no_shift_years))
    date = date - pd.to_timedelta(morning.astype(int), unit="D")
    value = rows.VALUE / np.where(rows.ELEMENT.isin(TENTHS), 10, 1)
    # A shifted high can land on a day that already has one. If the two differ we can't tell
    # which is right, so that day's value is left missing.
    both = rows.assign(DATE=date, VALUE=value).pivot_table(
        index="DATE", columns="ELEMENT", values="VALUE", aggfunc=["min", "max"])
    wide = both["min"].where(both["min"] == both["max"])
    wide = wide.reindex(index=CALENDAR, columns=ELEMENTS).astype("float32")
    # Range checks: some values pass NOAA's checks but are impossible for Texas.
    temps = ["TMAX", "TMIN"]
    low, high = TEXAS_RECORDS
    wide[temps] = wide[temps].where((wide[temps] >= low) & (wide[temps] <= high))
    # A low above the high means one of the two is wrong, and we can't tell which.
    wide.loc[wide.TMIN > wide.TMAX, temps] = np.nan
    wide.insert(0, "STATION", path.name[:11])
    return wide.reset_index()


def coverage(station, period):
    start, end = PERIODS[period]
    return station.TMAX[station.DATE.between(start, end)].notna().mean()


def build():
    """Clean every station, keep the well-covered ones, add targets and periods, and save."""
    if not any(RAW.glob("*.csv.gz")):
        import download_data
        download_data.main()
    # A partial download would silently drop stations and change which ones pass the coverage rule.
    if not STATIONS.exists():
        raise RuntimeError("data/stations.csv is missing: re-run python data_scripts/download_data.py")
    missing = set(pd.read_csv(STATIONS).ID) - {path.name[:11] for path in RAW.glob("*.csv.gz")}
    if missing:
        raise RuntimeError(f"{len(missing)} station files are missing: re-run python data_scripts/download_data.py")
    no_shift = pd.read_csv(NO_SHIFT_FILE, dtype={"STATION": str}).groupby("STATION").YEAR.agg(set)
    stations = [clean_station(path, no_shift.get(path.name[:11], set())) for path in sorted(RAW.glob("*.csv.gz"))]
    stations = [s for s in stations if min(coverage(s, "train"), coverage(s, "validation")) >= MIN_COVERAGE]
    data = pd.concat(stations, ignore_index=True)

    tmax = data.groupby("STATION", sort=False).TMAX
    for h in LEADS:
        data[f"y{h}"] = tmax.shift(-h)  # TMAX h days after the forecast day

    # A forecast day belongs to a period only if all 7 target days are inside it too. One more
    # day is kept back because a shifted high for a period's last day comes from a reading
    # filed the next morning, in the next period.
    data["period"] = pd.Series(pd.NA, index=data.index, dtype="string")
    for name, (start, end) in PERIODS.items():
        last = pd.Timestamp(end) - pd.Timedelta(days=max(LEADS) + 1)
        data.loc[data.DATE.between(start, last), "period"] = name
    # Every method needs today's TMAX and all 7 answers, so all are scored on the same rows.
    data["usable"] = data.period.notna() & data.TMAX.notna() & data[TARGETS].notna().all(axis=1)

    assert len(data) == len(stations) * len(CALENDAR), "a station is missing calendar days"
    assert not (data.TMIN > data.TMAX).any(), "a day has a low above its high"
    for name, (start, end) in PERIODS.items():
        days = data.DATE[data.period == name]
        assert days.min() >= pd.Timestamp(start) and days.max() + pd.Timedelta(days=8) <= pd.Timestamp(end)

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(TABLE)
    return data


def load_data(rebuild=False):
    """The cleaned table. Builds and saves it on the first call, or when rebuild=True."""
    if rebuild or not TABLE.exists():
        return build()
    return pd.read_pickle(TABLE)


if __name__ == "__main__":
    data = build()
    print(f"{data.STATION.nunique()} stations, {len(data):,} station-days, saved to {TABLE}")
    print(data[data.usable].period.value_counts().to_string())
