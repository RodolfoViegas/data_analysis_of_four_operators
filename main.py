import pandas as pd
import numpy as np
from itertools import combinations
import load_datasets
from stats_methods import StatsForecastMethods
from ml_methods import MLMethods
from combinators import Combinator


def main() -> None:
    # Loading  Time Series.
    monthly: dict = load_datasets.monthly_data()
    rmse_results: list = []
    smape_results: list = []
    writer = pd.ExcelWriter("model_predictions.xlsx", engine="xlsxwriter")

    # All time series were divided into two sets: train and test.
    key: str
    y_train: pd.Series
    y_test: pd.Series
    for key, (y_train, y_test) in monthly.items():
        freq: str = pd.infer_freq(y_train.index)
        y_train = y_train.asfreq(freq=freq)
        y_test = y_test.asfreq(freq=freq)

        # List of nine models used on experiemnts
        models: list = ["mlp", "knn", "svr", "arima", "ets", "theta", "tbats", "ces"]

        # The class StatsForecastMethods receives the series, the forecasting horizon and the seasonal period value
        # and returns, after the models be fitted, a dictionary which contains the fitted values and preds for all
        # statistical models
        stats_models: StatsForecastMethods = StatsForecastMethods(
            y_train=y_train, y_test=y_test, fh=1, sp=12
        )
        stats_models.fit()
        stats_results: dict = stats_models.results()

        # The class MLMethods receives the series, the forecasting horizon and the size of validation set,
        # which will be taken from train set at the time of trianing.
        # The class returns, after the all the three methods be fitted, a dictionary which contains the fitted values and preds for all
        # machine learing models
        ml_models: MLMethods = MLMethods(
            y_train=y_train, y_test=y_test, fh=1, validation_size=int(len(y_test) / 2)
        )
        ml_results: dict = ml_models.results()

        # All the fitted values and prediction are concatenated into a single pandas dataframe (maybe can be changed to faster Polars dataframe)
        #
        all_fitted: pd.DataFrame = pd.concat(
            [ml_results["fitted_values"], stats_results["fitted_values"]], axis=1
        )
        all_preds: pd.DataFrame = pd.concat(
            [ml_results["preds"], stats_results["preds"]], axis=1
        )

        # A general data fram that will accumulate all the fitted_values + prediction for all methods and their combinations.
        # The first and the second columns are columns with flags signaling that each observation can be part of train, validation and test set;
        # the second is the time series themselves
        series_data: pd.DataFrame = pd.DataFrame(
            {
                "using": ["train"] * (len(y_train) - len(y_test) // 2)
                + ["validation"] * (len(y_test) // 2)
                + ["test"] * len(y_test),
                "series": np.concatenate([y_train.values, y_test.values]),
            }
        )
        # Fuelling the General DataFrama, series_data, with the fitted_values + prediction of single models
        model: str
        for model in models:
            series_data[model] = np.concatenate(
                [all_fitted[model].values, all_preds[model].values]
            )

        num_models: int
        for num_models in range(len(models), 1, -1):
            # the function combination generates all possible combinations for the models list, the set function garantees no repeated tuples
            selected_models: tuple[str]
            for selected_models in set(combinations(models, num_models)):
                selected_fitted: pd.DataFrame = all_fitted[list(selected_models)]
                selected_preds: pd.DataFrame = all_preds[list(selected_models)]

                combinator: Combinator = Combinator(
                    fitted_values=selected_fitted,
                    y_test=y_test,
                    y_train=y_train,
                    preds=selected_preds,
                )

                mean_preds, mean_fitted_values = combinator.mean_method()
                median_preds, median_fitted_values = combinator.median_method()
                fi_avg_preds, fi_avg_fitted_values = combinator.fi_avg_method()
                stack_preds, stack_fitted_values = combinator.svr_stack_method()

                series_data[f"mean_{'_'.join(selected_models)}"] = np.concatenate(
                    [mean_fitted_values, mean_preds]
                )
                series_data[f"median_{'_'.join(selected_models)}"] = np.concatenate(
                    [median_fitted_values, median_preds]
                )
                series_data[f"fi_avg_{'_'.join(selected_models)}"] = np.concatenate(
                    [fi_avg_fitted_values, fi_avg_preds]
                )
                series_data[f"stack_{'_'.join(selected_models)}"] = np.concatenate(
                    [stack_fitted_values, stack_preds]
                )

                rmse_mean = compute_rmse(y_test, mean_preds)
                smape_mean = compute_smape(y_test, mean_preds)

                rmse_median = compute_rmse(y_test, median_preds)
                smape_median = compute_smape(y_test, median_preds)

                rmse_fi_avg = compute_rmse(y_test, fi_avg_preds)
                smape_fi_avg = compute_smape(y_test, fi_avg_preds)

                rmse_stack = compute_rmse(y_test, stack_preds)
                smape_stack = compute_smape(y_test, stack_preds)

                rmse_results.append(
                    {
                        "dataset": key,
                        "models_used": "_".join(selected_models),
                        "rmse_mean": rmse_mean,
                        "rmse_median": rmse_median,
                        "rmse_fi_avg": rmse_fi_avg,
                        "rmse_mode": rmse_stack,
                    }
                )

                smape_results.append(
                    {
                        "dataset": key,
                        "models_used": "_".join(selected_models),
                        "smape_mean": smape_mean,
                        "smape_median": smape_median,
                        "smape_fi_avg": smape_fi_avg,
                        "smape_mode": smape_stack,
                    }
                )

        series_data.to_excel(writer, sheet_name=key, index=False)

    rmse_results_df = pd.DataFrame(rmse_results)
    rmse_results_df.to_csv("model_sensitivity_rmse_results.csv", index=False)

    smape_results_df = pd.DataFrame(smape_results)
    smape_results_df.to_csv("model_sensitivity_smape_results.csv", index=False)

    writer.close()
    print("Excel and CSV files saved successfully!")


def compute_rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() ** 0.5


def compute_smape(y_true, y_pred):
    return (
        100
        * (
            np.absolute((y_true - y_pred))
            / ((np.absolute(y_true) + np.absolute(y_pred)) / 2)
        ).mean()
    )


if __name__ == "__main__":
    main()
