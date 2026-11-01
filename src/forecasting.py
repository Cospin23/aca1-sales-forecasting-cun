"""
Entrenamiento y pronóstico del modelo seleccionado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .features import RF_FEATURES, make_features, weekly_sales


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def optimize_random_forest(train: pd.DataFrame):
    search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [200, 500],
            "max_depth": [5, 10, None],
            "max_features": [0.7, 1.0],
            "min_samples_leaf": [1, 2, 4],
        },
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )
    search.fit(train[RF_FEATURES], train["Sales"])
    return search


def recursive_forecast(weekly: pd.DataFrame, model, horizon: int = 4) -> pd.DataFrame:
    history = weekly[["Date", "Sales"]].copy()
    history["Date"] = pd.to_datetime(history["Date"])

    output = []
    for _ in range(horizon):
        f = make_features(history)
        latest = f.iloc[-1:][RF_FEATURES]
        pred = float(model.predict(latest)[0])
        next_date = history["Date"].iloc[-1] + pd.Timedelta(weeks=1)

        output.append({"Date": next_date, "Predicted_Sales": pred})
        history = pd.concat(
            [history, pd.DataFrame({"Date": [next_date], "Sales": [pred]})],
            ignore_index=True,
        )

    return pd.DataFrame(output)
