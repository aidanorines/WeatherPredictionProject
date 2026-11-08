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

2. Download the data. This saves one file per station to `data_scripts/data/raw/`, about 104 MB in total:

   ```bash
   python data_scripts/download_data.py
   ```

3. Check the download. This prints the station, value, and element counts used in the proposal:

   ```bash
   python data_scripts/dataset_summary.py
   ```

The `data/` folder is not tracked by git.

## Folders

* `ProjectProposal/`: the project proposal, as Markdown and PDF
* `data_scripts/`: scripts that download and summarize the data

## Branches

* `main`: working code that both of us use
* `project-setup`: data cleaning, scoring, and repo setup
* `random-forest` and `gradient-boosting`: one branch per model, created from `main` after setup is merged

Merge into `main` when a piece works.