# WeatherPredictionProject

Forecasting daily high temperature 1 to 7 days ahead at Texas weather stations using NOAA GHCN-Daily data (2000-2025). Random forest baseline, gradient boosting improvement. CS4347 project, Texas State.

## Team

* Aidan Orines: data cleaning, scoring, and the gradient boosting model
* Danylo Huk: the random forest model

## Setup

You need Python 3.12 or newer.

1. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Download the data. This saves one file per station to `data_scripts/data/raw/`, about 104 MB in total, and each station's location to `data_scripts/data/stations.csv`:

   ```bash
   python data_scripts/download_data.py
   ```

3. Check the download. This prints the station, value, and element counts used in the proposal:

   ```bash
   python data_scripts/dataset_summary.py
   ```

The `data/` folder is not tracked by git.

4. Optional: check the date alignment of COOP stations against nearby airports. This prints the evidence for shifting morning TMAX back a day and leaving TMIN unshifted. It also rewrites `data_scripts/no_shift.csv`, the station-years whose morning highs are already on the right day, so rebuild the table afterwards with `python data_scripts/prepare_data.py`:

   ```bash
   python data_scripts/check_date_alignment.py
   ```

## Folders

* `ProjectProposal/`: the project proposal, as Markdown and PDF
* `data_scripts/`: scripts that download and summarize the data

## Branches

* `main`: working code that both of us use
* `project-setup`: data cleaning, scoring, and repo setup
* `random-forest` and `gradient-boosting`: one branch per model, created from `main` after setup is merged

Merge into `main` when a piece works.