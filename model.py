from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from tqdm import tqdm
from datetime import date
from datetime import datetime, date

from sklearn.linear_model import QuantileRegressor


def split_train_test(
    df: pd.DataFrame,
    test_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets.

    The last `test_days` days are held out for testing.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        test_days (int): The number of days to include in the test set (default: 30).
            use ">=" sign for df_test

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
        A tuple containing the train and test DataFrames.
    """

    def f(x):
        return (date.today() - x.to_pydatetime().date()).days

    new_df = df.copy()
    new_df["days"] = new_df["day"].apply(f)
    last_day = new_df["days"].values[-1]
    df_train, df_test = (
        df.loc[new_df["days"] > (last_day + test_days)],
        df.loc[new_df["days"] <= (last_day + test_days)],
    )
    return df_train, df_test


class MultiTargetModel:
    def __init__(
        self,
        features: List[str],
        horizons: List[int] = [7, 14, 21],
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> None:
        """
        Parameters
        ----------
        features : List[str]
            List of features columns.
        horizons : List[int]
            List of horizons.
        quantiles : List[float]
            List of quantiles.

        Attributes
        ----------
        fitted_models_ : dict
            Dictionary with fitted models for each sku_id.
            Example:
            {
                sku_id_1: {
                    (quantile_1, horizon_1): model_1,
                    (quantile_1, horizon_2): model_2,
                    ...
                },
                sku_id_2: {
                    (quantile_1, horizon_1): model_3,
                    (quantile_1, horizon_2): model_4,
                    ...
                },
                ...
            }

        """
        self.quantiles = quantiles
        self.horizons = horizons
        self.sku_col = "sku_id"
        self.date_col = "day"
        self.features = features
        self.targets = [f"next_{horizon}d" for horizon in self.horizons]

        self.fitted_models_ = {}

    def fit(self, data: pd.DataFrame, verbose: bool = False) -> None:
        """Fit model on data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit on.
        verbose : bool, optional
            Whether to show progress bar, by default False
            Optional to implement, not used in grading.
        """
        for num, val in enumerate(data.groupby(self.sku_col)):
            d = {}
            for i, hor in enumerate(self.horizons):
                for q in self.quantiles:
                    qr = QuantileRegressor(quantile=q, solver="highs")
                    train = val[1].dropna(subset=self.features)
                    train = train.dropna(subset=[self.targets[i]])
                    qr.fit(
                        train.loc[:, self.features],
                        train.loc[:, self.targets[i]],
                    )

                    d[(q, hor)] = qr
            sku_num = f"sku_id_{num}"
            self.fitted_models_[sku_num] = d

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict on data.

        Predict 0 values for a new sku_id.

        Parameters
        ----------
        data : pd.DataFrame
            Data to predict on.

        Returns
        -------
        pd.DataFrame
            Predictions.
        """
        res = pd.DataFrame()
        for num, val in enumerate(data.groupby(self.sku_col)):
            temp = pd.DataFrame()
            temp[self.sku_col] = val[1][self.sku_col]
            temp[self.date_col] = val[1][self.date_col]
            for i, hor in enumerate(self.horizons):
                for q in self.quantiles:
                    sku_num = f"sku_id_{num}"
                    if sku_num not in self.fitted_models_:
                        column = f"pred_{hor}d_q{int(q * 100)}"
                        temp[column] = [0] * len(val[1])
                    else:
                        model = self.fitted_models_[sku_num][(q, hor)]
                        pred = model.predict(val[1].loc[:, self.features])
                        column = f"pred_{hor}d_q{int(q * 100)}"
                        pred = np.nan_to_num(pred)
                        temp[column] = pred

            res = pd.concat([res, temp])

        return res


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Calculate the quantile loss between the true and predicted values.

    The quantile loss measures the deviation between the true
        and predicted values at a specific quantile.

    Parameters
    ----------
    y_true : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    quantile : float
        The quantile to calculate the loss for.

    Returns
    -------
    float
        The quantile loss.
    """
    if len(y_true) == 0:
        return 0.0
    if y_true.shape != y_pred.shape:
        return 0.0
    y_true = np.nan_to_num(y_true)
    y_pred = np.nan_to_num(y_pred)
    loss = quantile * np.maximum(y_true - y_pred, np.zeros(y_true.shape)) + (
        1 - quantile
    ) * np.maximum(y_pred - y_true, np.zeros(y_true.shape))
    return np.nanmean(loss)


# def evaluate_model(
#     df_true: pd.DataFrame,
#     df_pred: pd.DataFrame,
#     quantiles: List[float] = [0.1, 0.5, 0.9],
#     horizons: List[int] = [7, 14, 21],
# ) -> pd.DataFrame:
#     """Evaluate model on data.

#     Parameters
#     ----------
#     df_true : pd.DataFrame
#         True values.
#     df_pred : pd.DataFrame
#         Predicted values.
#     quantiles : List[float], optional
#         Quantiles to evaluate on, by default [0.1, 0.5, 0.9].
#     horizons : List[int], optional
#         Horizons to evaluate on, by default [7, 14, 21].

#     Returns
#     -------
#     pd.DataFrame
#         Evaluation results.
#     """
#     losses = {}

#     for quantile in quantiles:
#         for horizon in horizons:
#             true = df_true[f"next_{horizon}d"].values
#             pred = df_pred[f"pred_{horizon}d_q{int(quantile*100)}"].values
#             loss = quantile_loss(true, pred, quantile)

#             losses[(quantile, horizon)] = loss

#     #print(losses)
#     losses = pd.DataFrame(losses, index=["loss"]).T.reset_index()
#     losses.columns = ["quantile", "horizon", "avg_quantile_loss"]  # type: ignore

#     return losses

# data = pd.read_csv("res.csv")

# df_train, df_test = split_train_test(data)

# model = MultiTargetModel(
#     features=[
#         "price",
#         "qty",
#         "qty_7d_avg",
#         "qty_7d_q10",
#         "qty_7d_q50",
#         "qty_7d_q90",
#         "qty_14d_avg",
#         "qty_14d_q10",
#         "qty_14d_q50",
#         "qty_14d_q90",
#         "qty_21d_avg",
#         "qty_21d_q10",
#         "qty_21d_q50",
#         "qty_21d_q90",
#     ],
#     horizons=[7, 14, 21],
#     quantiles=[0.1, 0.5, 0.9],
# )
# model.fit(df_train, verbose=True)

# predictions = model.predict(df_test)

# print(predictions)

# print(evaluate_model(df_test, predictions))
