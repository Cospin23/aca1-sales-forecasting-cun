from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.features import RF_FEATURES, CYCLIC_FEATURES, load_raw, weekly_sales, make_features

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "train.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

df = load_raw(str(DATA))
weekly = weekly_sales(df)

basic = make_features(weekly, include_cyclic=False)
cyclic = make_features(weekly, include_cyclic=True)

basic_model = basic.dropna(subset=RF_FEATURES + ["Sales"]).reset_index(drop=True)
cyclic_model = cyclic.dropna(subset=CYCLIC_FEATURES + ["Sales"]).reset_index(drop=True)

train = basic_model.iloc[:125]
test = basic_model.iloc[125:]
train_c = cyclic_model.iloc[:125]
test_c = cyclic_model.iloc[125:]

def metrics(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
    }

# Baseline: última semana observada.
baseline = metrics(test["Sales"], test["lag_1"])

# Regresión lineal con variables cíclicas.
lr = LinearRegression().fit(train_c[CYCLIC_FEATURES], train_c["Sales"])
lr_metrics = metrics(test_c["Sales"], lr.predict(test_c[CYCLIC_FEATURES]))

# Random Forest inicial.
rf_initial = RandomForestRegressor(
    n_estimators=200, random_state=42, n_jobs=-1
).fit(train[RF_FEATURES], train["Sales"])
rf_initial_metrics = metrics(
    test["Sales"], rf_initial.predict(test[RF_FEATURES])
)

# Gradient Boosting clásico.
gb = GradientBoostingRegressor(random_state=42).fit(
    train[RF_FEATURES], train["Sales"]
)
gb_metrics = metrics(test["Sales"], gb.predict(test[RF_FEATURES]))

# Optimización temporal de Random Forest.
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

optimized_metrics = metrics(
    test["Sales"], search.best_estimator_.predict(test[RF_FEATURES])
)

# Modelo final: se reentrena con todas las observaciones disponibles.
final_model = RandomForestRegressor(
    **search.best_params_, random_state=42, n_jobs=-1
)
final_model.fit(basic_model[RF_FEATURES], basic_model["Sales"])

joblib.dump(final_model, MODELS / "sales_forecasting_rf.pkl")

metadata = {
    "dataset_rows": int(len(df)),
    "dataset_columns": int(df.shape[1]),
    "postal_code_nulls": int(df["Postal Code"].isna().sum()),
    "exact_duplicates": int(df.duplicated().sum()),
    "date_min": str(df["Order Date"].min().date()),
    "date_max": str(df["Order Date"].max().date()),
    "weekly_rows": int(len(weekly)),
    "model_rows": int(len(basic_model)),
    "train_rows": 125,
    "test_rows": 32,
    "baseline": baseline,
    "linear_regression_cyclical": lr_metrics,
    "random_forest_initial": rf_initial_metrics,
    "gradient_boosting": gb_metrics,
    "random_forest_optimized": optimized_metrics,
    "random_forest_cv_mae": float(-search.best_score_),
    "best_params": search.best_params_,
    "rf_features": RF_FEATURES,
    "cyclic_features": CYCLIC_FEATURES,
}

(MODELS / "metadata.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Modelo final generado:", MODELS / "sales_forecasting_rf.pkl")
print(json.dumps(metadata, indent=2, ensure_ascii=False))
