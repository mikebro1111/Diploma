import multiprocessing
from typing import Callable, Optional

from pandas import DataFrame
import os

import pandas as pd

from data_service import DataService
from fileutils import get_sub_folders, get_files
from market_analyzer import AssetMarketAnalyzer
from stockdata import StockDataProcessor


def extract_dataframe(files, folder, read_func: Callable):
    return pd.concat((read_func(os.path.join(folder, file)) for file in files),
                     ignore_index=True)


def get_folder_dataframe(folder: str) -> Optional[DataFrame]:
    csv_files = get_files(folder, '.csv')

    if csv_files:
        df = extract_dataframe(csv_files, folder, pd.read_csv)
        return df if not df.empty else None

    json_files = get_files(folder, '.json')

    if json_files:
        df = extract_dataframe(json_files, folder, pd.read_json)
        return df if not df.empty else None


def merge_dfs(results):
    valid_results = filter(lambda result: not not result, results)
    if not valid_results:
        return

    combined_df = pd.concat(valid_results, ignore_index=True)
    return combined_df


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
    root_folder = r'./dataset'  # Змінив на відносний шлях
    folders = get_sub_folders(root_folder)
    with multiprocessing.Pool() as pool:
        results = pool.map(get_folder_dataframe, folders)
    combined_df = merge_dfs(results)

    if combined_df:
        print(combined_df)

    stock_data_processor = StockDataProcessor()

    data_service = DataService()

    market_analyzer = AssetMarketAnalyzer('<FILEPATH>', stock_data_processor.sp500, stock_data_processor.predictors)

# TODO: Finish this file and move some logic. Write tests


if __name__ == '__main__':
    main()
