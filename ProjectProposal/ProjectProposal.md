# Project Proposal Title: Forecasting Daily High Temperature Over the Next Seven Days at Texas Weather Stations

## Teammates

Group name: HODA

* Aidan Orines, hgn24, data cleaning, scoring, and the gradient boosting model
* Danylo Huk, rnj42, the random forest model

## Project Abstract

We will predict the daily high temperature for the next 7 days at Texas weather stations. We will use NOAA GHCN-Daily records from 2000 to 2025. We will use the past to predict the future, so we will not use cross-validation.

We will start with a random forest as our baseline. Then we will try to beat it with gradient boosting and better features.

## Problem Statement

**Problem.** Given a station's observations through day t, predict its TMAX in °C for each of days t+1 to t+7.

**Importance.** Daily high temperature affects everyday decisions in Texas. It matters for heat safety, outdoor work, and power and water use. We want to see how far ahead past station data can help predict it.

**Benchmark.** We will compare every model against three simple forecasts built from the training years. Persistence says the coming days will match today's high. Climatology uses the usual high for that date at that station. Damped anomaly starts with the usual high and adds part of how far today is from normal. Lead time just means how many days ahead the forecast is. We are using all three because each one is strong at different lead times. Persistence is hard to beat at short lead times. Climatology is hard to beat at long lead times.

**Data.** The data come from NOAA's Global Historical Climatology Network Daily (GHCN-Daily). Details are in the Dataset section.

**Informal success measures.** We will call it a success if a model has a lower MAE in °C than the best simple forecast at each lead time. Every method will use the same stations and days. MAE is the average size of the error, so it tells us how many degrees we are off. We will also report RMSE, which counts big misses more. We will also report the lead time where any improvement disappears.

**What we hope to achieve.** We hope to learn how well past station data can predict the high temperature 1 to 7 days out. We also want to find the lead time where it stops beating simple forecasts.

## Dataset

**Source and format.** GHCN-Daily version 3.35-upd-2026110217, downloaded on 2026-11-04 from NOAA NCEI (`https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/`) by `download_data.py`. The download covers Texas stations with a TMAX record overlapping 2000-2025. It keeps every element those stations report. Each station has one gzipped CSV in long format, with one row per station, date, and element. The columns are `ID`, `DATE`, `ELEMENT`, `VALUE`, `MFLAG`, `QFLAG`, `SFLAG`, and `OBS_TIME`. The dataset is not from Kaggle and contains no images, video, or text.

**Size.** `dataset_summary.py` produced these counts from the downloaded files:

| Measure | Count |
|---|---:|
| Stations | 674 |
| Observed station-days | 4,475,952 |
| Raw values | 29,505,579 |
| Values passing NOAA quality checks | 29,409,998 |
| Passing values, excluding weather-type flags | 28,707,988 |
| Element types with passing values | 78 |

The dataset exceeds the 10 million float requirement. The stations comprise 474 cooperative (COOP), 148 WBAN, and 52 RAWS stations.

**Main attributes.** Ranges below are for values passing NOAA quality checks, converted from stored units.

| Element | Description | Range | Values | Stations |
|---|---|---|---:|---:|
| TMAX | Daily maximum temperature | -18.9 to 48.3 °C | 4,214,312 | 674 |
| TMIN | Daily minimum temperature | -28.3 to 36.7 °C | 4,189,880 | 672 |
| PRCP | Precipitation | 0 to 661.2 mm | 3,925,802 | 617 |
| SNOW | Snowfall | 0 to 635 mm | 2,786,364 | 597 |
| SNWD | Snow depth | 0 to 508 mm | 2,762,200 | 595 |
| TOBS | Temperature at observation time | -26.1 to 44.4 °C | 2,758,247 | 476 |
| AWND | Average wind speed | 0 to 49.6 m/s | 860,963 | 139 |
| WSF2 | Fastest 2-minute wind speed | 0 to 111.8 m/s | 836,433 | 132 |
| WDF2 | Direction of fastest 2-minute wind | 0 to 360° | 836,213 | 132 |
| TAVG | Average temperature, method varies | -40.8 to 56.6 °C | 584,965 | 126 |
| RHAV | Average relative humidity | 3 to 100% | 251,921 | 37 |
| ADPT | Average dew point | -26.1 to 27.2 °C | 250,779 | 37 |
| ASLP | Average sea-level pressure | 988.8 to 1045.0 hPa | 250,221 | 37 |
| WT and WV types | Weather-type flags, such as thunder and fog | 1 or blank | 702,010 | Varies |

The remaining elements are sparser.

**Why this data fits.** This data has TMAX, the value we want to predict. It also has other weather elements we can use as inputs. It covers 26 years across many Texas stations, so we can train on older years and test on newer ones. The main limit is that stations have gaps. Not every station reports every element.

**Known characteristics.**

* The 26 years contain 9,497 calendar days. Passing TMAX values cover 65.8% of the 6,400,978 possible station-days.
* Elements are stored in different units. For example, TMAX is in tenths of °C, while SNOW is in whole mm.
* Values that fail quality checks remain in the files, with a letter in `QFLAG`.
* Days with no report have no row.
* 61.2% of passing TMAX values have an observation time before 12:00.
* Element availability varies widely: TMAX exists at 674 stations, and relative humidity at 37.
* Some values pass NOAA's checks but look wrong for Texas, such as TAVG at -40.8 °C and 56.6 °C.

## Methodology

**Data split.** We will split the data by year so we only use the past to predict the future. We will not use cross-validation because it would mix future days into training. A real forecast can only use what already happened. We will train on 2000-2019. We will use 2020-2022 to pick our models and settings. We will report final results on 2023-2025.

| Period | Years |
|---|---|
| Training | 2000-2019 |
| Validation | 2020-2022 |
| Testing | 2023-2025 |

**Baseline and improvement tools.** Our baseline will be a random forest on past station observations. To improve on it, we will use gradient boosting with scikit-learn's HistGradientBoosting. We will also use feature engineering, feature selection, and dimensionality reduction such as PCA. The inputs may include other weather elements, more past days, and data from nearby stations. We will compare everything against the baseline on the validation years.

**Related work.** Shalaby and Embark used GHCN-Daily data from 23 Egyptian stations. They compared LSTM, XGBoost, random forest, and other models for daily average temperature. Singh and Rawat compared XGBoost, SVM, and random forest on 45 years of daily temperatures from one airport. They found no clear winner. We did not find a paper with our exact setup, so we will start with tree models.

## Teaming Strategy

Aidan will clean the data and set up the scoring that both models use. Then Aidan will own the gradient boosting model, and Danylo will own the random forest. Each of us will tune our own model and test improvements on it. We will review each other's work.

We will meet every Friday. We are using our MS Teams group chat for now. We will set up a GitHub repository for our code.

### Role Assignments and Commitment Matrix

| Target | Task | Owner |
|---|---|---|
| Nov 13 | Cleaned data, date alignment, scoring code, simple forecasts | Aidan |
| Nov 20 | First random forest result | Danylo |
| Nov 20 | First gradient boosting result | Aidan |
| Nov 20 | Project Progress submitted | Both |
| Nov 27 | Feature and dimensionality comparisons, fallbacks if behind | Aidan for gradient boosting, Danylo for random forest |
| Dec 1, 11:00 AM | Test results and slides | Both |
| Dec 4 | Final write-up and code | Both |

## Mitigation Plan

**Data alternative.** If we can't fix the date or gap problems at the cooperative stations, we will evaluate on airport stations only. We will keep the rest of the data as inputs.

**Task alternative.** If the longer lead times show no improvement, we will focus on lead times 1 to 3. We will report lead times 4 to 7 as the limit of what station data can do.

**Method alternative.** If the random forest is too slow on the full data, we will train on a sample of stations or switch to Extra Trees. If gradient boosting doesn't beat the random forest, we will keep the random forest and explain why.

**If a teammate is MIA.** If one of us misses a Friday meeting without saying anything, the other will reach out that day. If there is still no response, that person will take over the critical work and cut the scope to one model.

**If the baseline is GIGO.** If a model can't beat the simple forecasts, we will check the date alignment and quality flags first. Then we will change the features or scope.
