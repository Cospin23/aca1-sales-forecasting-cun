"""
Feature engineering del proyecto ACA 1.
La serie temporal se construye semanalmente mediante resample("W").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RF_FEATURES = [
    "year", "month", "quarter", "weekofyear",
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_13", "lag_26", "lag_52",
    "rolling_mean_4", "rolling_mean_8", "rolling_mean_13",
]

CYCLIC_FEATURES = [
    "year", "month", "quarter", "weekofyear",
    "month_sin", "month_cos", "week_sin", "week_cos",
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_13", "lag_26", "lag_52",
    "rolling_mean_4", "rolling_mean_8", "rolling_mean_13",
]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    return df


def weekly_sales(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["Order Date", "Sales"]).copy()
    weekly = (
        work.set_index("Order Date")["Sales"]
        .resample("W")
        .sum()
        .reset_index()
        .rename(columns={"Order Date": "Date"})
    )
    return weekly.sort_values("Date").reset_index(drop=True)


def make_features(weekly: pd.DataFrame, include_cyclic: bool = False) -> pd.DataFrame:
    d = weekly.copy().sort_values("Date").reset_index(drop=True)

    d["year"] = d["Date"].dt.year
    d["month"] = d["Date"].dt.month
    d["quarter"] = d["Date"].dt.quarter
    d["weekofyear"] = d["Date"].dt.isocalendar().week.astype(int)

    for lag in [1, 2, 4, 8, 13, 26, 52]:
        d[f"lag_{lag}"] = d["Sales"].shift(lag)

    for window in [4, 8, 13]:
        # shift(1) evita que el valor objetivo de la semana actual
        # entre en la media móvil usada para predecirla.
        d[f"rolling_mean_{window}"] = (
            d["Sales"].shift(1).rolling(window).mean()
        )

    if include_cyclic:
        d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
        d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
        d["week_sin"] = np.sin(2 * np.pi * d["weekofyear"] / 52)
        d["week_cos"] = np.cos(2 * np.pi * d["weekofyear"] / 52)

    return d
