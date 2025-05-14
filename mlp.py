import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
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


def mlp_regressor_iterative(
    y_train: pd.Series, y_test: pd.Series, fh: int, validation_size: int
):
    regressor = MLPRegressor(
        n_iter_no_change=5,
        early_stopping=True,
        shuffle=False,
    )

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
        "forecaster__estimator__hidden_layer_sizes": [20, 50, 100],
        "forecaster__estimator__learning_rate_init": np.logspace(-5, -1, 15),
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

