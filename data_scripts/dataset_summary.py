"""Print the dataset facts quoted in ProjectProposal.md. Run after download_data.py."""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"
CALENDAR_DAYS = len(pd.date_range("2000-01-01", "2025-12-31"))


def summarize(path):
    """Return raw row count, per-element stats and TMAX observation times for one station file."""
    rows = pd.read_csv(path, usecols=["DATE", "ELEMENT", "VALUE", "QFLAG", "OBS_TIME"],
                       dtype={"DATE": "int32", "VALUE": "float32", "OBS_TIME": "str"}, low_memory=False)
    passed = rows[rows.QFLAG.isna()]  # blank QFLAG = passed all checks
    elements = passed.groupby("ELEMENT").VALUE.agg(["count", "min", "max"])
    tmax = passed[passed.ELEMENT == "TMAX"]
    return {"station": path.name[:11], "rows": len(rows), "days": rows.DATE.nunique(), "elements": elements,
            "morning": (tmax.OBS_TIME.dropna().astype(int) < 1200).sum(), "tmax": len(tmax)}


def element_table(stations):
    parts = [s["elements"].assign(station=s["station"]) for s in stations]
    table = pd.concat(parts).groupby(level=0).agg(
        values=("count", "sum"), stations=("station", "nunique"), low=("min", "min"), high=("max", "max"))
    return table.sort_values("values", ascending=False)


def main():
    stations = [summarize(path) for path in sorted(RAW.glob("*.csv.gz"))]
    table = element_table(stations)
    flags = table.index.str.match(r"W[TV]\d\d")  # weather-type flags are 1 or blank, not measurements
    tmax = sum(s["tmax"] for s in stations)
    possible = len(stations) * CALENDAR_DAYS
    print(f"stations {len(stations):,}; station-days {sum(s['days'] for s in stations):,}")
    print(f"raw values {sum(s['rows'] for s in stations):,}; passing {table['values'].sum():,}; "
          f"elements {len(table)}")
    print(f"passing non-flag values {table['values'][~flags].sum():,}; passing weather flags {table['values'][flags].sum():,}")
    print(f"station types {pd.Series([s['station'][:3] for s in stations]).value_counts().to_dict()}")
    print(f"calendar days {CALENDAR_DAYS:,}; passing TMAX coverage {tmax / possible:.1%} of {possible:,} station-days")
    print(f"TMAX with morning observation time {sum(s['morning'] for s in stations) / tmax:.1%}")
    print(table.to_string())


if __name__ == "__main__":
    main()
