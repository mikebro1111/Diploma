from datetime import datetime

import numpy as np
import pandas as pd
import os
import multiprocessing
import yfinance as yf
import dateutil.parser
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

from Data.datas.fileutils import get_sub_folders

PREDICTORS = ["Close", "Volume", "Open", "High", "Low"]

COLS = [
    'Global Bonds (Unhedged)',
    'Total US Bond Market',
    'US Large Cap Growth',
    'US Large Cap Value',
    'US Small Cap Growth',
    'US Small Cap Value',
    'Emerging Markets',
    'Intl Developed ex-US Market',
    'Short Term Treasury'
]

ASSET_WEIGHTS_CSV = '.\\data\\asset_weights.csv'

RETURNS_CSV = '.\\data\\asset_returns.csv'


class DataProcessor:
    @staticmethod
    def process_folder(folder):
        csv_files = [file for file in os.listdir(folder) if file.endswith('.csv')]

        if csv_files:
            df = pd.concat((pd.read_csv(os.path.join(folder, file)) for file in csv_files), ignore_index=True)
            return df if not df.empty else None

        json_files = [file for file in os.listdir(folder) if file.endswith('.json')]

        if json_files:
            df = pd.concat((pd.read_json(os.path.join(folder, file)) for file in json_files), ignore_index=True)
            return df if not df.empty else None

    @staticmethod
    def merge_dfs(results: DataFrame):
        valid_results = filter(lambda result: not result.empty, results)
        if not valid_results:
            return

        combined_df = pd.concat(valid_results, ignore_index=True)
        return combined_df


class FinancialDataProcessor:
    @staticmethod
    def read_json_data(file_path):
        return pd.read_json(file_path)

    @staticmethod
    def read_sp500_data():
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        return sp500


    @staticmethod
    def preprocess_sp500_data(sp500_data):
        sp500_data.index = pd.to_datetime(sp500_data.index)

        sp500_data.plot.line(y="Close", use_index=True)

        del sp500_data["Dividends"]
        del sp500_data["Stock Splits"]

        sp500_data["Tomorrow"] = sp500_data["Close"].shift(-1)

        sp500_data["Target"] = (sp500_data["Tomorrow"] > sp500_data["Close"]).astype(int)

        sp500_data = sp500_data.loc["1990-01-01":].copy()

        return sp500_data


class MachineLearningModel:
    @staticmethod
    def train_random_forest_model(train_data, predictors):
        model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
        model.fit(train_data[predictors], train_data["Target"])
        return model

    @staticmethod
    def evaluate_model(model, test_data, predictors):
        preds = model.predict(test_data[predictors])
        preds = pd.Series(preds, index=test_data.index)
        precision_score(test_data["Target"], preds)
        return preds


class FinancialAnalysis:
    @staticmethod
    def calculate_global_market_return(asset_returns, asset_weights, treasury_rate):
        excess_asset_returns = asset_returns.subtract(treasury_rate, axis=0)
        global_return = excess_asset_returns.mean().multiply(asset_weights['weight'].values).sum()
        return global_return

    @staticmethod
    def calculate_risk_aversion(asset_returns, asset_weights, treasury_rate):
        excess_asset_returns = asset_returns.subtract(treasury_rate, axis=0)
        cov_matrix = excess_asset_returns.cov()
        global_return = FinancialAnalysis.calculate_global_market_return(asset_returns, asset_weights, treasury_rate)
        market_var = np.matmul(asset_weights.values.reshape(len(asset_weights)).T,
                               np.matmul(cov_matrix.values, asset_weights.values.reshape(len(asset_weights))))
        risk_aversion = global_return / market_var
        return risk_aversion


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)


def main():
    root_folder = os.path.normpath('.\dataset')

    folders = get_sub_folders(root_folder)
    with multiprocessing.Pool() as pool:
        results = pool.map(DataProcessor.process_folder, folders)

    combined_df = DataProcessor.merge_dfs(results)

    if combined_df is not None:
        print(combined_df)

    read_last_date_df = str(combined_df['startDate'].values[-1])
    last_date_df = dateutil.parser.isoparse(read_last_date_df)
    today = datetime.today()
    dif_day_dates = today - last_date_df

    if dif_day_dates.days > 0:
        combined_df.to_json('test.csv')

    sp500 = FinancialDataProcessor.read_sp500_data()
    sp500 = FinancialDataProcessor.preprocess_sp500_data(sp500)

    train_data = sp500.iloc[:-100]
    test_data = sp500.iloc[-100:]

    model = MachineLearningModel.train_random_forest_model(train_data, PREDICTORS)
    precision = MachineLearningModel.evaluate_model(model, test_data, PREDICTORS)

    predictions = backtest(sp500, model, PREDICTORS)

    precision_score(predictions["Target"], predictions["Predictions"])

    predictions["Target"].value_counts() / predictions.shape[0]

    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []

    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

        new_predictors += [ratio_column, trend_column]

    sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])

    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

    predictions = backtest(sp500, model, new_predictors)

    predictions["Predictions"].value_counts()

    precision_score(predictions["Target"], predictions["Predictions"])

    predictions["Target"].value_counts() / predictions.shape[0]

    filename = '../data.csv'

    sp500.to_csv(filename, index=False)

    print(f'Successfully saved DataFrame to {filename}.')

    # combined = pd.concat([test_data["Target"], PREDICTORS], axis=1)
    # combined.plot()

    asset_returns_orig = pd.read_csv(RETURNS_CSV, index_col='Year', parse_dates=True)
    asset_weights = pd.read_csv(ASSET_WEIGHTS_CSV, index_col='asset_class')

    asset_returns = asset_returns_orig[COLS].dropna()
    treasury_rate = asset_returns['Short Term Treasury']
    asset_returns = asset_returns[COLS[:-1]].astype(np.float_).dropna()
    asset_weights = asset_weights.loc[COLS[:-1]]

    global_return = FinancialAnalysis.calculate_global_market_return(asset_returns, asset_weights, treasury_rate)
    print("Global market return:", global_return)

    risk_aversion = FinancialAnalysis.calculate_risk_aversion(asset_returns, asset_weights, treasury_rate)
    print("Risk aversion parameter:", risk_aversion)


if __name__ == '__main__':
    main()
