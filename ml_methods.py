import rf, mlp, svr, knn
import pandas as pd


class MLMethods:
    def __init__(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        validation_size: int,
        fh: int = 18,
    ) -> None:
        self.y_train: pd.Series = y_train
        self.y_test: pd.Series = y_test
        self.validation_size: int = validation_size
        self.fh: range = fh

    def __str__(self):
        return f"{self.method_name}"

    def results(self):
        columns = ["rf","mlp", "knn", "svr"]

        RF: dict = rf.rf_regressor_iterative(
            y_train=self.y_train,
            y_test=self.y_test,
            fh=self.fh,
            validation_size=self.validation_size,
        )

        MLP: dict = mlp.mlp_regressor_iterative(
            y_train=self.y_train,
            y_test=self.y_test,
            fh=self.fh,
            validation_size=self.validation_size,
        )
        KNN: dict = knn.knn_regressor_iterative(
            y_train=self.y_train,
            y_test=self.y_test,
            fh=self.fh,
            validation_size=self.validation_size,
        )

        SVR: dict = svr.svr_regressor_iterative(
            y_train=self.y_train,
            y_test=self.y_test,
            fh=self.fh,
            validation_size=self.validation_size,
        )
        
        preds = pd.concat(
            [
                RF['preds'],
                MLP["preds"],
                KNN["preds"],
                SVR["preds"],
            ],
            axis=1,
        )
        
        preds.columns = columns

        fitted_values = pd.concat(
            [
                RF['fitted_values'],
                MLP["fitted_values"],
                KNN["fitted_values"],
                SVR["fitted_values"],
            ],
            axis=1,
        )

        fitted_values.columns = columns

        best_params = {
            "rf": RF['best_params'],
            "mlp": MLP["best_params"],
            "knn": KNN["best_params"],
            "svr": SVR["best_params"],
        }

        model = {
            "rf": RF['model'],
            "mlp": MLP["model"],
            "knn": KNN["model"],
            "svr": SVR["model"],
        }

        results = {
            "best_params":best_params,
            "best_params": best_params,
            "preds": preds,
            "fitted_values": fitted_values,
            "model": model,
        }

        return results
