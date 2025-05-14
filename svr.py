import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

from sktime.forecasting.compose import (
    RecursiveReductionForecaster,
    DirectReductionForecaster,
)
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV
from sktime.forecasting.base import ForecastingHorizon
from sktime.split import ExpandingWindowSplitter, SingleWindowSplitter
from sktime.transformations.series.impute import Imputer


def svr_regressor_iterative(
    y_train: pd.Series, y_test: pd.Series, fh: int, validation_size: int
):
    forecaster = TransformedTargetForecaster(
        steps=[
            ("scaler", TabularToSeriesAdaptor(MinMaxScaler())),
            (
                "forecaster",
                RecursiveReductionForecaster(
                    window_length=12,
                    estimator=SVR(),
                ),
            ),
            ("imputer", Imputer()),
        ]
    )

    param_grid = {
        "forecaster__estimator__C": [10, 100, 1000],
        "forecaster__estimator__gamma": [0.1, 0.01, 0.001],
        "forecaster__estimator__epsilon": [0.1, 0.01, 0.001],
    }

    validation_size = validation_size
    cv = SingleWindowSplitter(
        window_length=len(y_train) - validation_size, fh=range(1, validation_size + 1)
    )

    gscv = ForecastingRandomizedSearchCV(
        forecaster=forecaster,
        param_distributions=param_grid,
        n_iter=30,
        cv=cv,
        error_score="raise",
    )
    gscv.fit(y=y_train)
    first = gscv.predict(fh=1)
    preds = gscv.update_predict(y=y_test, update_params=False)

    preds = pd.concat([first, preds])
    fitted_values = gscv.predict(fh=-np.arange(len(y_train)))

    results = {
        "best_params": gscv.best_params_,
        "preds": preds,
        "fitted_values": fitted_values,
        "model": gscv.best_forecaster_,
    }

    return results

