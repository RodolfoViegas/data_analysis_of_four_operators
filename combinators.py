import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import gaussian_kde
from sklearn.svm import SVR


class Combinator:
    def __init__(
        self,
        fitted_values: pd.DataFrame,
        y_test: pd.Series,
        preds: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        self.fitted_values: pd.Series = fitted_values
        self.y_test: pd.Series = y_test
        self.preds: pd.DataFrame = preds
        self.y_train: pd.Series = y_train

    def __str__(self):
        return "Combinator"

    def mean_method(self):
        mean_preds = self.preds.mean(axis=1)
        mean_preds.index = self.y_test.index  # [1:]

        mean_fitted_values = self.fitted_values.mean(axis=1)
        mean_fitted_values.index = self.y_train.index

        return mean_preds, mean_fitted_values

    def median_method(self):
        mean_preds = self.preds.median(axis=1)
        mean_preds.index = self.y_test.index  # [1:]

        median_fitted_values = self.fitted_values.median(axis=1)
        median_fitted_values.index = self.y_train.index

        return mean_preds, median_fitted_values

    def fi_avg_method(self, jitter=1e-6):
        # estimattion of feature importances weights
        trees = ExtraTreesRegressor()
        trees.fit(X=self.fitted_values, y=self.y_train)
        weights = trees.feature_importances_

        fi_avg_preds = pd.Series(self.preds.values.dot(weights))
        fi_avg_preds.index = self.y_test.index  # [1:]

        fi_avg_fitted_values = pd.Series(self.fitted_values.values.dot(weights))
        fi_avg_fitted_values.index = self.y_train.index

        return fi_avg_preds, fi_avg_fitted_values
    
    def svr_stack_method(self):

        param_list={'C': [10, 100, 1000],  
                'gamma': [0.1, 0.01, 0.001],  
                'epsilon': [0.1, 0.01, 0.001] 
                }

        ts = TimeSeriesSplit(3)
        svr = SVR()

        search = RandomizedSearchCV(
              estimator=svr,
              param_distributions=param_list,
              n_iter=30,
              n_jobs=-1,
              cv=3,
              verbose=0,
              scoring='neg_mean_squared_error',
              )

       
        search.fit(X=self.fitted_values, y=self.y_train)

        stack_preds = pd.Series(search.predict(self.preds.values))
        stack_preds.index = self.y_test.index

        stack_fitted_values = pd.Series(search.predict(self.fitted_values.values))
        stack_fitted_values.index = self.y_train.index

        return stack_preds, stack_fitted_values

    def mode_method(self, jitter=1e-6):
        mode_forecasts = []

        for _, row in self.preds.iterrows():
            forecasts = row.values.astype(float)

            # If all values are identical, return that value directly
            if np.all(forecasts == forecasts[0]):
                mode_value = forecasts[0]
            else:
                forecasts += np.random.normal(
                    0, jitter, size=forecasts.shape
                )  # Add jitter
                kde = gaussian_kde(forecasts)
                x_vals = np.linspace(min(forecasts), max(forecasts), 1000)
                density_vals = kde(x_vals)
                mode_value = x_vals[np.argmax(density_vals)]

            mode_forecasts.append(mode_value)

        mode_preds = pd.Series(np.array(mode_forecasts))
        mode_preds.index = self.y_test.index  # [1:]

        ##############################################

        mode_fitted_values = []

        for _, row in self.fitted_values.iterrows():
            forecasts = row.values.astype(float)

            # If all values are identical, return that value directly
            if np.all(forecasts == forecasts[0]):
                mode_value = forecasts[0]
            else:
                forecasts += np.random.normal(
                    0, jitter, size=forecasts.shape
                )  # Add jitter
                kde = gaussian_kde(forecasts)
                x_vals = np.linspace(min(forecasts), max(forecasts), 1000)
                density_vals = kde(x_vals)
                mode_value = x_vals[np.argmax(density_vals)]

            mode_fitted_values.append(mode_value)

        mode_fitted_values = pd.Series(np.array(mode_fitted_values))
        mode_fitted_values.index = self.y_train.index

        return mode_preds, mode_fitted_values


"""
def mode_ensemble_rowwise(df, jitter=1e-6):
    mode_forecasts = []

    for _, row in df.iterrows():
        forecasts = row.values.astype(float)

        # If all values are identical, return that value directly
        if np.all(forecasts == forecasts[0]):
            mode_value = forecasts[0]
        else:
            forecasts += np.random.normal(0, jitter, size=forecasts.shape)  # Add jitter
            kde = gaussian_kde(forecasts)
            x_vals = np.linspace(min(forecasts), max(forecasts), 1000)
            density_vals = kde(x_vals)
            mode_value = x_vals[np.argmax(density_vals)]

        mode_forecasts.append(mode_value)

    return np.array(mode_forecasts)




    mode_forecasts = []

        for _, row in self.preds.iterrows():
            forecasts = row.values  # Convert row to a NumPy array
            kde = gaussian_kde(forecasts)
            x_vals = np.linspace(min(forecasts), max(forecasts), 1000)
            density_vals = kde(x_vals)
            mode_value = x_vals[np.argmax(density_vals)]
            mode_forecasts.append(mode_value)
        
        mode_preds = pd.Series(np.array(mode_forecasts))
        mode_preds.index = self.y_test.index[1:]

        return mode_forecasts


"""
