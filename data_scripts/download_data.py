"""Download NOAA GHCN-Daily rows for Texas stations with TMAX in 2000-2025.

Usage:
    python download_data.py list    # print the station count, download nothing
    python download_data.py         # download into data/raw/, safe to re-run

Saves one long-format file per station: one row per station, date and element.
Also saves data/stations.csv: each station's name, latitude, longitude and elevation (m).
Do not swap in the AWS by_station copies: 200 Texas stations there stopped in February 2025.
"""
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"
STATIONS = DATA / "stations.csv"
COLS = ["ID", "DATE", "ELEMENT", "VALUE", "MFLAG", "QFLAG", "SFLAG", "OBS_TIME"]


def texas_stations():
    """Texas by state code, with a TMAX year range overlapping the window, and each station's location."""
    stations = pd.read_fwf(BASE + "ghcnd-stations.txt",
                           colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71)],
                           names=["ID", "LATITUDE", "LONGITUDE", "ELEVATION", "STATE", "NAME"],
                           dtype={"ID": str, "STATE": str, "NAME": str})
    inventory = pd.read_fwf(BASE + "ghcnd-inventory.txt", colspecs=[(0, 11), (31, 35), (36, 40), (41, 45)],
                            names=["ID", "ELEMENT", "FIRST", "LAST"], dtype={"ID": str, "ELEMENT": str})
    tmax = inventory[(inventory.ELEMENT == "TMAX") & (inventory.FIRST <= 2025) & (inventory.LAST >= 2000)]
    texas = stations[(stations.STATE == "TX") & stations.ID.isin(tmax.ID)].sort_values("ID")
    assert len(texas) > 600, f"expected about 674 stations, got {len(texas)}"
    return texas.drop(columns="STATE")


def main():
    stations = texas_stations()
    ids = list(stations.ID)
    print(f"{len(ids)} Texas stations", flush=True)
    if sys.argv[1:] == ["list"]:
        return
    RAW.mkdir(parents=True, exist_ok=True)
    # Latitude, longitude and elevation, for nearest-station inputs and level features.
    stations.to_csv(STATIONS, index=False)
    # NOAA updates values over time, so keep the exact version for the report.
    # Only a fresh download records it: a re-run fills gaps and keeps the files it already has.
    if not any(RAW.glob("*.csv.gz")):
        urllib.request.urlretrieve(BASE + "ghcnd-version.txt", DATA / "ghcnd-version.txt")
    failed = []
    for n, station in enumerate(ids, 1):
        target = RAW / f"{station}.csv.gz"
        if target.exists():
            continue
        part = RAW / f"{station}.part"  # a stopped run never leaves a half file under the final name
        try:
            rows = pd.read_csv(f"{BASE}by_station/{station}.csv.gz", header=None, names=COLS, dtype=str)
            rows = rows[rows.DATE.between("20000101", "20251231")]
            rows.to_csv(part, index=False, compression="gzip")
            part.replace(target)
            print(f"{n}/{len(ids)} {station} {len(rows)} rows", flush=True)
        except (OSError, EOFError, ValueError) as error:
            failed.append(station)
            print(f"{n}/{len(ids)} {station} failed: {error}", flush=True)
    print(f"done, {len(failed)} failed" + (", re-run to retry" if failed else ""), flush=True)


if __name__ == "__main__":
    main()
