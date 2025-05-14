from sktime.forecasting.statsforecast import (
    StatsForecastAutoARIMA,
    StatsForecastAutoCES,
    StatsForecastAutoTBATS,
    StatsForecastAutoETS,
    StatsForecastAutoTheta,
)

from sktime.forecasting.base import ForecastingHorizon
import pandas as pd
import numpy as np


class StatsForecastMethods:
    def __init__(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        sp: int,
        fh: int = 18,
    ) -> None:
        self.y_train: pd.Series = y_train
        self.y_test: pd.Series = y_test
        self.sp: int = sp
        self.fh: range = range(1, fh + 1)
        self.method_dict: dict = {
            "arima": StatsForecastAutoARIMA(sp=self.sp),
            "ets": StatsForecastAutoETS(season_length=self.sp),
            "ces": StatsForecastAutoCES(season_length=self.sp),
            "theta": StatsForecastAutoTheta(season_length=self.sp),
            "tbats": StatsForecastAutoTheta(season_length=self.sp),
        }

    def __str__(self):
        return f"{self.method_name}"

    def fit(self) -> None:
        name: str
        for name in self.method_dict:
            self.method_dict[name].fit(y=self.y_train, fh=self.fh)

    def predict(self):
        preds: dict = {}
        for name in self.method_dict:
            first = self.method_dict[name].predict(fh=1)
            preds[name] = self.method_dict[name].update_predict(
                y=self.y_test, update_params=False
            )
            preds[name] = pd.concat([first, preds[name]])
        df_preds = pd.DataFrame(preds)

        return df_preds

    def fitted_values(self):
        fitted_values: dict = {}
        for name in self.method_dict:
            fitted_values[name] = self.method_dict[name].predict(
                -np.arange(len(self.y_train))
            )

        df_fitted_values = pd.DataFrame(fitted_values)

        return df_fitted_values

    def results(self):
        results = {
            "best_params": {
                name: self.method_dict[name].get_fitted_params()
                for name in self.method_dict
            },
            "preds": self.predict(),
            "fitted_values": self.fitted_values(),
            "model": self.method_dict,
        }

        return results


def arima(y_train: pd.Series, y_test: pd.Series, sp: int, fh: int):
    forecaster = StatsForecastAutoARIMA(sp=sp)

    forecaster.fit(y=y_train, fh=range(1, fh + 1))

    preds = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))

    fitted_values = forecaster.predict(fh=-np.arange(len(y_train)))

    results = {
        "best_params": forecaster.get_param_names(),
        "preds": preds,
        "fitted_values": fitted_values,
        "model": forecaster,
    }

    return results


def ets(y_train: pd.Series, y_test: pd.Series, sp: int, fh: int):
    forecaster = StatsForecastAutoETS(season_length=sp)

    forecaster.fit(y=y_train, fh=range(1, fh + 1))

    preds = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))

    fitted_values = forecaster.predict(fh=-np.arange(len(y_train)))

    results = {
        "best_params": forecaster.get_param_names(),
        "preds": preds,
        "fitted_values": fitted_values,
        "model": forecaster,
    }

    return results


def ces(y_train: pd.Series, y_test: pd.Series, sp: int, fh: int):
    forecaster = StatsForecastAutoCES(season_length=sp)

    forecaster.fit(y=y_train, fh=range(1, fh + 1))

    preds = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))

    fitted_values = forecaster.predict(fh=-np.arange(len(y_train)))

    results = {
        "best_params": forecaster.get_param_names(),
        "preds": preds,
        "fitted_values": fitted_values,
        "model": forecaster,
    }

    return results


def tbats(y_train: pd.Series, y_test: pd.Series, sp: int, fh: int):
    forecaster = StatsForecastAutoTBATS(seasonal_periods=sp)

    forecaster.fit(y=y_train, fh=range(1, fh + 1))

    preds = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))

    fitted_values = forecaster.predict(fh=-np.arange(len(y_train)))

    results = {
        "best_params": forecaster.get_param_names(),
        "preds": preds,
        "fitted_values": fitted_values,
        "model": forecaster,
    }

    return results


def theta(y_train: pd.Series, y_test: pd.Series, sp: int, fh: int):
    forecaster = StatsForecastAutoTheta(season_length=sp)

    forecaster.fit(y=y_train, fh=range(1, fh + 1))

    preds = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))

    fitted_values = forecaster.predict(fh=-np.arange(len(y_train)))

    results = {
        "best_params": forecaster.get_param_names(),
        "preds": preds,
        "fitted_values": fitted_values,
        "model": forecaster,
    }

    return results
