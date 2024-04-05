import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


class AssetMarketAnalyzer:

    def __init__(self, file_path, sp500, predictors):
        self.file_path = file_path
        self.sp500 = sp500
        self.model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
        self.predictors = predictors

        horizons = [2, 5, 60, 250, 1000]
        self.new_predictors = self.generate_new_predictors(horizons)

        self.configure_asset_returns()

        self.global_return = None
        self.market_var = None
        self.risk_aversion = None

        self.predictions = self.backtest()

    def generate_new_predictors(self, horizons):
        new_predictors = []

        for horizon in horizons:
            rolling_averages = self.sp500['Close'].rolling(horizon).mean()

            ratio_column = f"Close_Ratio_{horizon}"
            self.sp500[ratio_column] = self.sp500["Close"] / rolling_averages

            trend_column = f"Trend_{horizon}"
            self.sp500[trend_column] = self.sp500["Target"].shift(1).rolling(horizon).sum()

            new_predictors += [ratio_column, trend_column]

        return new_predictors

    def configure_asset_returns(self):
        cols = ['Global Bonds (Unhedged)', 'Total US Bond Market', 'US Large Cap Growth',
                     'US Large Cap Value', 'US Small Cap Growth', 'US Small Cap Value', 'Emerging Markets',
                     'Intl Developed ex-US Market', 'Short Term Treasury']

        asset_returns_orig = pd.read_csv('asset_returns.csv', index_col='Year', parse_dates=True)

        self.asset_weights = pd.read_csv('asset_weights.csv', index_col='asset_class')
        self.asset_returns = asset_returns_orig[cols].dropna()
        self.treasury_rate = self.asset_returns['Short Term Treasury']

        self.asset_returns.mean()

    def backtest(self):
        return self.model.predict(self.sp500)

    def compute_precision_score(self, true_labels, predicted_labels):
        return precision_score(true_labels, predicted_labels)

    def compute_mean_returns(self):
        return self.asset_returns.mean()

    def compute_global_market_stats(self):
        excess_asset_returns = self.asset_returns.subtract(self.treasury_rate, axis=0)
        cov = excess_asset_returns.cov()
        self.global_return = excess_asset_returns.mean().multiply(self.asset_weights['weight'].values).sum()

        self.market_var = np.matmul(self.asset_weights.values.reshape(len(self.asset_weights)).T,
                               np.matmul(cov.values, self.asset_weights.values.reshape(len(self.asset_weights))))

        self.risk_aversion = self.global_return / self.market_var

        return self.global_return, self.market_var, self.risk_aversion

    def save_data(self, filename):
        self.sp500.to_csv(filename, index=False)
        print(f'Successfully saved DataFrame to {filename}.')
