"""Check which day a COOP station's TMAX and TMIN belong to, using the nearest airport.

    python check_date_alignment.py

Airports (USW) that give no reading time, or 2400, record midnight-to-midnight
days, so their dates are right. Some USW stations are read in the morning like COOP
stations (Longview WSMO, Clarendon, Palestine 2 NE), so they are never used as the
reference. A COOP observer reads the thermometer once a day and reports the high and
low of the last 24 hours. For each COOP station in the cleaned table, this compares
its raw values with the nearest midnight airport within 50 km on the day before, the
same day and the day after. The offset with the smallest gap is the day the value
really belongs to: -1 means it belongs to the day before the date it is filed under.
Only training years are used, so no cleaning choice is made from validation or test data.

It then repeats the TMAX test for every station and year, because observers change
practice: many stations already filed the high on the right day in the early 2000s and
needed the shift later. Station-years where the unshifted high matches better are written
to no_shift.csv, which prepare_data.py reads. This scan uses training and validation years,
since validation years are for choices. For test years, each station keeps the majority
decision of its last 3 checked years, so test data is never used.
"""
import numpy as np
import pandas as pd

from download_data import STATIONS
from prepare_data import COOP_SOURCES, NO_SHIFT_FILE, PERIODS, RAW, load_data

MAX_KM = 50
MIN_DAYS = 365  # fewer matched days than this is too noisy to judge a station
OFFSETS = [-1, 0, 1]
START, END = PERIODS["train"]
CHECK_END = PERIODS["validation"][1]  # the year-by-year scan may use validation years, never test years
YEAR_KM = 75  # NOAA's own timing check compares with stations within 75 km
YEAR_DAYS = 180  # matched days needed to judge one station-year


def km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def read(station, element, end=END):
    """Raw passing values in °C from 2000 to `end`, filed under their original date, with the reading time."""
    rows = pd.read_csv(RAW / f"{station}.csv.gz",
                       usecols=["DATE", "ELEMENT", "VALUE", "QFLAG", "SFLAG", "OBS_TIME"],
                       dtype={"DATE": str, "OBS_TIME": str, "SFLAG": str}, low_memory=False)
    rows = rows[rows.QFLAG.isna() & (rows.ELEMENT == element)
                & rows.DATE.between(START.replace("-", ""), end.replace("-", ""))]
    hour = pd.to_numeric(rows.OBS_TIME, errors="coerce")
    reading = np.select([hour < 1200, hour >= 1200], ["morning", "afternoon/evening"], "no time given")
    return pd.DataFrame({"DATE": pd.to_datetime(rows.DATE, format="%Y%m%d"), "VALUE": rows.VALUE / 10,
                         "READING": reading, "HOUR": hour, "SOURCE": rows.SFLAG})


def midnight(rows):
    """Only values from a midnight-to-midnight day: no reading time given, or read at 2400."""
    return rows[rows.HOUR.isna() | (rows.HOUR == 2400)]


def midnight_airports():
    """Midnight TMAX of each USW station that is a valid reference, by date."""
    stations = pd.read_csv(STATIONS).set_index("ID")
    usw = stations.index[stations.index.str.startswith("USW")]
    # Longview WSMO files 9% of its highs at 2400 and the rest at 0900, so a station must be
    # midnight almost all the time to serve as the reference.
    tmax = {a: read(a, "TMAX", CHECK_END) for a in usw}
    return {a: midnight(rows).set_index("DATE").VALUE for a, rows in tmax.items()
            if len(rows) >= MIN_DAYS and len(midnight(rows)) >= 0.9 * len(rows)}


def pairs(airports):
    """Each COOP station in the cleaned table with its nearest midnight airport, if one is within MAX_KM."""
    stations = pd.read_csv(STATIONS).set_index("ID")
    coop = [s for s in load_data().STATION.unique() if s.startswith("USC")]
    airports = stations.loc[list(airports)]
    for station in coop:
        lat, lon = stations.loc[station, ["LATITUDE", "LONGITUDE"]]
        distance = km(lat, lon, airports.LATITUDE, airports.LONGITUDE)
        if distance.min() <= MAX_KM:
            yield station, distance.idxmin(), distance.min()


def gaps(station, airport, element):
    """Mean |COOP - airport| for each reading time and offset, per station."""
    coop, ap = read(station, element), midnight(read(airport, element)).set_index("DATE").VALUE
    out = []
    for offset in OFFSETS:
        match = ap.reindex(coop.DATE + pd.Timedelta(days=offset)).to_numpy()
        gap = (coop.VALUE - match).abs()
        for reading, g in gap.groupby(coop.READING.to_numpy()):
            out.append({"STATION": station, "ELEMENT": element, "READING": reading,
                        "OFFSET": offset, "GAP": g.mean(), "DAYS": g.notna().sum()})
    return out


def yearly(airports):
    """Gap to the nearest midnight airport, shifted and unshifted, for each station-year of morning COOP TMAX."""
    stations = pd.read_csv(STATIONS).set_index("ID")
    out = []
    for station in load_data().STATION.unique():
        x = read(station, "TMAX", CHECK_END)
        x = x[(x.READING == "morning") & x.SOURCE.isin(COOP_SOURCES)]
        if x.empty:
            continue
        lat, lon = stations.loc[station, ["LATITUDE", "LONGITUDE"]]
        near = sorted((km(lat, lon, *stations.loc[a, ["LATITUDE", "LONGITUDE"]]), a) for a in airports)
        for year, g in x.groupby(x.DATE.dt.year):
            for distance, a in near:  # the nearest airport with enough matched days that year
                if distance > YEAR_KM:
                    break
                shifted = (g.VALUE - airports[a].reindex(g.DATE - pd.Timedelta(days=1)).to_numpy()).abs()
                unshifted = (g.VALUE - airports[a].reindex(g.DATE).to_numpy()).abs()
                both = shifted.notna() & unshifted.notna()
                if both.sum() >= YEAR_DAYS:
                    out.append({"STATION": station, "YEAR": year, "AIRPORT": a, "KM": round(distance),
                                "DAYS": both.sum(), "SHIFTED": shifted[both].mean(),
                                "UNSHIFTED": unshifted[both].mean()})
                    break
    return pd.DataFrame(out)


def no_shift_years(scan):
    """Checked years where the unshifted high matches better, plus every test year for
    stations whose last 3 checked years mostly matched better unshifted."""
    better = scan.UNSHIFTED < scan.SHIFTED  # full precision; an exact tie keeps the shift
    recent = better.groupby(scan.STATION).apply(lambda b: b.tail(3).mean() > 0.5)  # scan is sorted by year
    later = range(int(CHECK_END[:4]) + 1, int(PERIODS["test"][1][:4]) + 1)
    carried = pd.DataFrame([{"STATION": s, "YEAR": y} for s in recent[recent].index for y in later])
    keep = pd.concat([scan[better], carried], ignore_index=True)
    return keep.astype({"KM": "Int64", "DAYS": "Int64"})  # carried rows have no evidence


def main():
    airports = midnight_airports()
    found = list(pairs(airports))
    print(f"{len(found)} COOP stations have an airport within {MAX_KM} km "
          f"(median {np.median([d for *_, d in found]):.0f} km)\n")
    rows = pd.DataFrame([r for s, a, _ in found for e in ["TMAX", "TMIN"] for r in gaps(s, a, e)])
    rows = rows[rows.DAYS >= MIN_DAYS]
    wide = rows.pivot_table(index=["ELEMENT", "READING", "STATION"], columns="OFFSET", values="GAP").dropna()
    best = wide.idxmin(axis=1)
    summary = wide.groupby(level=["ELEMENT", "READING"]).mean().round(2)
    summary.columns = [f"gap {o:+d} day (°C)" for o in summary.columns]
    for offset in OFFSETS:
        summary[f"stations best at {offset:+d}"] = (best == offset).groupby(level=["ELEMENT", "READING"]).sum()
    print("Average gap to the nearest airport, by where the value is moved (mean over stations):")
    print(summary.to_string())

    scan = yearly(airports)
    keep = no_shift_years(scan)
    keep.to_csv(NO_SHIFT_FILE, index=False, float_format="%.2f")  # rounded for display only
    checked = keep.SHIFTED.notna()
    print(f"\nYear by year: {scan.STATION.nunique()} stations with morning COOP highs, "
          f"{len(scan)} station-years checked, {checked.sum()} match better unshifted.")
    print(f"{(~checked).sum()} test station-years carried forward. Saved to {NO_SHIFT_FILE.name}.")
    names = pd.read_csv(STATIONS).set_index("ID").NAME
    for station, g in keep.groupby("STATION"):
        print(f"  {station} {names[station]:26s} {' '.join(str(y) for y in g.YEAR)}")


if __name__ == "__main__":
    main()
