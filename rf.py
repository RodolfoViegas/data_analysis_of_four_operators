import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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


def rf_regressor_iterative(
    y_train: pd.Series, y_test: pd.Series, fh: int, validation_size: int
):
    regressor = RandomForestRegressor()

    forecaster = TransformedTargetForecaster(
        steps=[
            ("scaler", TabularToSeriesAdaptor(MinMaxScaler())),
            (
                "forecaster",
                RecursiveReductionForecaster(
                    window_length=12,
                    estimator=regressor,
                ),
            ),
            ("imputer", Imputer()),
        ]
    )

    param_grid = {
        "forecaster__estimator__n_estimators": [50, 100, 200], 
        "forecaster__estimator__max_depth": [5, 10],
        "forecaster__estimator__max_features": [0.6, 0.8, 1.0],
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
