# Volatility Forecasting Framework

COS 4091 Senior Project, American University in Bulgaria, Fall 2026.
Author: Presian Valentinov Mihaylov. Supervisor: Prof. Gabriela Mogos.

The project aims to forecast the volatility of 25 assets (equities, ETFs, crypto, currencies, commodity futures) at hourly and daily
frequency with 10 models (GARCH, ARIMA and HAR families) and 3 blends, using walk-forward one-step forecasts.
This is visualized using a Streamlit dashboard, which has further features, such as model recommendation.


## Libraries
The libraris used in the project are listed below:

numpy, pandas - data handling and model computations
scipy - numerical optimization
yfinance, requests - downloading the raw data
streamlit, plotly - dashboard
pytest - tests
arch, statsmodels - model validation


## Run
Python 3.13 (developed with 3.13.4).
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py            starts the dashboard
python -m pytest          in order to run the tests
```
All data and results the dashboard needs are included.



## Structure
```
App/            Streamlit dashboard
Blending/       The three blends and their evaluation
Data/           Download, validate and feature deriving scripts, datasets, results
Forecasting/    Walk-forward forecasting procedure
Models/         10 models, validation section as well
tests/          tests
```

## How to rerun the whole pipeline
From the project root, in this order:
```
python Data/Data_Gathering/<script>.py       overwrites the included raw data
python Data/Data_Validation/data_existance.py
python Data/Data_Validation/data_value_validation.py
python "Data/Data Modeling/derive_daily_features.py"
python "Data/Data Modeling/derive_intraday_features.py"
python -m Forecasting.run_for                currently it takes significant amount of time, so not recommended
python -m Blending.run_blends
python -m Blending.eval
```

Importantly, the blending results may vary slightly when re-running. However, the difference is extremely marginal.