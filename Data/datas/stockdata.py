import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier


class StockDataProcessor:

    def __init__(self):
        self.predictors = ["Close", "Volume", "Open", "High", "Low"]

        self.sp500 = self.download_sp500_data()
        self.sp500 = self.prepare_sp500(self.sp500)

        train_data = self.sp500.iloc[:-100]
        test = self.sp500.iloc[-100:]

        model = self.train_model(train_data, self.predictors)

        preds = self.backtest(test, model, self.predictors)

        combined = pd.concat([test["Target"], preds], axis=1)
        combined.plot()

    def download_sp500_data(self):
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        sp500.to_csv("sp500.csv")
        return sp500

    def prepare_sp500(self, sp500):
        sp500.index = pd.to_datetime(sp500.index)
        sp500.plot.line(y="Close", use_index=True)
        del sp500["Dividends"]
        del sp500["Stock Splits"]
        sp500["Tomorrow"] = sp500["Close"].shift(-1)
        sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
        return sp500.loc["1990-01-01":].copy()

    def train_model(self, train, predictors):
        model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
        model.fit(train[predictors], train["Target"])
        return model

    def backtest(self, data, model, predictors):
        predictions = model.predict(data[predictors])
        predictions = pd.Series(predictions, index=data.index)
        return predictions

