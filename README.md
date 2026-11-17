# WeatherPredictionProject

Forecasting daily high temperature 1 to 7 days ahead at Texas weather stations using NOAA GHCN-Daily data (2000-2025). Random forest baseline, gradient boosting improvement. CS4347 project, Texas State.

## Team

* Aidan Orines: data cleaning, scoring, and the gradient boosting model
* Danylo Huk: the random forest model

## Setup

You need Python 3.12 or newer. Run every command from the repository root.

1. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Download the data. This saves one file per station to `data_scripts/data/raw/`, about 104 MB in total, and each station's location to `data_scripts/data/stations.csv`. It is safe to re-run: it only fetches missing files.

   ```bash
   python data_scripts/download_data.py
   ```

3. Check the download. This prints the station, value, and element counts used in the proposal. Your counts can differ slightly, because the proposal used NOAA's Nov 4 release; `data_scripts/data/ghcnd-version.txt` records the release you downloaded.

   ```bash
   python data_scripts/dataset_summary.py
   ```

4. Build the cleaned table. This saves `data_scripts/data/processed/stations.pkl` and prints the station and row counts. The first `load_data()` call in a notebook does the same thing.

   ```bash
   python data_scripts/prepare_data.py
   ```

5. Optional: check the date alignment of COOP stations against nearby airports. This prints the evidence for moving morning TMAX back a day. Its TMIN check is a tie, so leaving TMIN on its filed date is a modeling choice, not a finding. It also rewrites `data_scripts/no_shift.csv`, the station-years whose morning highs are already on the right day, so rebuild the table afterwards with step 4:

   ```bash
   python data_scripts/check_date_alignment.py
   ```

The `data/` folder is not tracked by git, so each of us downloads and builds it locally.

## The cleaned table

One row per station and calendar day, from 2000 to 2025.

* **Stations:** a station is kept if it has a TMAX on at least 80% of days in the training years (2000-2019) and in the validation years (2020-2022). 280 of the 674 downloaded stations pass.
* **Targets:** `y1` to `y7` hold the TMAX 1 to 7 days after the row's date. `period` is `train`, `validation` or `test`.
* **Usable rows:** `usable` marks rows with today's TMAX and all 7 targets. Every method is scored on these rows, and `score()` rejects any other row.
* **Forecast time:** COOP observers report the previous afternoon's high in the morning, so those highs are moved back a day. A row's forecast uses data through its date, day t, so it is issued the morning of day t+1, and `y1` is that same day's high.
* **Precipitation and snow:** values NOAA marks as "missing, presumed zero" (`MFLAG` P) are kept as 0. That is about a third of the SNOW and SNWD values and 8% of the PRCP values.

## Training a model

Each model is trained in a Jupyter notebook in `notebooks/` that imports the shared scripts:

```python
import sys
sys.path.insert(0, "../data_scripts")  # relative to the notebook's folder

from prepare_data import load_data
from scoring import simple_forecasts, score

data = load_data()
train = data[data.usable & (data.period == "train")]
rows = data[data.usable & (data.period == "validation")]

# Fit one model per lead time on `train`, with targets y1 to y7.
# predictions: one row per row of `rows`, one column per lead time, y1 to y7 in order.
score(rows, predictions)  # MAE and RMSE in °C for each lead time

forecasts = simple_forecasts(data, rows)  # persistence, climatology, damped anomaly
score(rows, forecasts["damped anomaly"])  # the simple forecast to beat
```

* Notebooks need `ipykernel`. VS Code offers to install it the first time a cell runs.
* Restart the kernel after editing a shared script, or the notebook keeps the old version.
* Import the scripts this way, not as `data_scripts.prepare_data`, because they import each other by bare name.

## Folders

* `ProjectProposal/`: the project proposal, as Markdown and PDF
* `data_scripts/`: scripts that download, clean and score the data, and check the date alignment
* `notebooks/`: one training notebook per model, added on the model branches

## Branches

* `main`: working code that both of us use
* `project-setup`: data cleaning, scoring, and repo setup
* `random-forest` and `gradient-boosting`: one branch per model, created from `main` after setup is merged

Merge into `main` when a piece works.
