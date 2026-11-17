"""Shared scoring and simple forecasts, so every model is judged the same way.

    from prepare_data import load_data
    from scoring import simple_forecasts, score

    data = load_data()
    rows = data[data.usable & (data.period == "validation")]
    forecasts = simple_forecasts(data, rows)  # fitted on training years only
    score(rows, forecasts["damped anomaly"])

A model's predictions go into score() the same way: one column per lead time,
in the order y1 to y7, one row per row of `rows`.
"""
import numpy as np
import pandas as pd

LEADS = range(1, 8)
TARGETS = [f"y{h}" for h in LEADS]


def score(rows, predictions):
    """MAE and RMSE in °C for each lead time."""
    # Checks run before any arithmetic, since NumPy would silently repeat a short array across rows.
    if len(rows) == 0:
        raise ValueError("no rows to score")
    if isinstance(predictions, pd.DataFrame) and not (predictions.index.equals(rows.index)
                                                      and list(predictions.columns) == TARGETS):
        raise ValueError("prediction frame must have the same rows as `rows` and columns y1 to y7")
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(rows), len(TARGETS)):
        raise ValueError(f"need predictions of shape {(len(rows), len(TARGETS))}, got {predictions.shape}")
    if not np.isfinite(predictions).all():
        raise ValueError("every prediction must be a finite number")
    error = predictions - rows[TARGETS].to_numpy(dtype=float)
    if np.isnan(error).any():
        raise ValueError("score only usable rows: some targets are missing")
    return pd.DataFrame({"MAE": np.abs(error).mean(axis=0), "RMSE": np.sqrt((error ** 2).mean(axis=0))},
                        index=pd.Index(LEADS, name="lead time"))


def _seasons(dates):
    """A smooth yearly cycle: a constant plus 3 sine and cosine pairs."""
    t = 2 * np.pi * dates.dt.dayofyear.to_numpy() / 365.25
    return np.column_stack([np.ones_like(t)] + [f(k * t) for k in (1, 2, 3) for f in (np.sin, np.cos)])


def climatology(data):
    """A function giving each station's usual TMAX on any date, fitted on training years only."""
    train = data[(data.period == "train") & data.TMAX.notna()]
    coefs = pd.DataFrame({station: np.linalg.lstsq(_seasons(rows.DATE), rows.TMAX.to_numpy(float), rcond=None)[0]
                          for station, rows in train.groupby("STATION")}).T
    return lambda stations, dates: (_seasons(dates) * coefs.loc[stations.to_numpy()].to_numpy()).sum(axis=1)


def simple_forecasts(data, rows):
    """Persistence, climatology and damped anomaly forecasts for `rows`, as arrays of shape (rows, 7)."""
    usual = climatology(data)
    future = np.column_stack([usual(rows.STATION, rows.DATE + pd.Timedelta(days=h)) for h in LEADS])
    today = rows.TMAX.to_numpy(float)
    anomaly = today - usual(rows.STATION, rows.DATE)

    # Share of today's departure from normal that is still there h days later, fitted on training rows.
    train = data[data.usable & (data.period == "train")]
    a0 = train.TMAX.to_numpy(float) - usual(train.STATION, train.DATE)
    share = np.array([(a0 * (train[f"y{h}"].to_numpy(float) - usual(train.STATION, train.DATE + pd.Timedelta(days=h)))).sum()
                      / (a0 ** 2).sum() for h in LEADS])

    return {"persistence": np.repeat(today[:, None], len(TARGETS), axis=1),
            "climatology": future,
            "damped anomaly": future + anomaly[:, None] * share}
