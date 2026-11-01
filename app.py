from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import RF_FEATURES, load_raw, weekly_sales
from src.forecasting import recursive_forecast

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "train.csv"
MODEL = ROOT / "models" / "sales_forecasting_rf.pkl"
META = ROOT / "models" / "metadata.json"

st.set_page_config(
    page_title="Sales Forecasting | CUN",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Sales Forecasting — Retail Superstore")
st.caption(
    "ACA 1 · Aprendizaje Automático · Especialización en Inteligencia Artificial · CUN"
)

if not DATA.exists():
    st.error("No se encontró data/train.csv.")
    st.stop()
if not MODEL.exists() or not META.exists():
    st.error("No se encontró el modelo entrenado. Ejecute: python train_model.py")
    st.stop()

@st.cache_data
def get_weekly():
    return weekly_sales(load_raw(str(DATA)))

@st.cache_resource
def get_model():
    return joblib.load(MODEL)

@st.cache_data
def get_metadata():
    return json.loads(META.read_text(encoding="utf-8"))

weekly = get_weekly()
model = get_model()
metadata = get_metadata()

with st.sidebar:
    st.header("🔮 Pronóstico")
    horizon = st.selectbox(
        "Horizonte de predicción",
        [1, 2, 4, 8],
        index=2,
        help="Número de semanas futuras que serán estimadas.",
    )
    run = st.button("Generar pronóstico", type="primary")

    st.divider()
    st.subheader("Modelo")
    st.write("Random Forest Regressor")
    st.write(f"Árboles: {metadata['best_params']['n_estimators']}")
    st.write(f"Profundidad máxima: {metadata['best_params']['max_depth']}")
    st.write(f"MAE CV: ${metadata['random_forest_cv_mae']:,.2f}")

c1, c2, c3 = st.columns(3)
c1.metric("Registros", f"{metadata['dataset_rows']:,}")
c2.metric("Semanas", f"{metadata['weekly_rows']:,}")
c3.metric("MAE test", f"${metadata['random_forest_optimized']['MAE']:,.2f}")

tab1, tab2, tab3 = st.tabs(["📈 Histórico y pronóstico", "🤖 Evaluación", "ℹ️ Metodología"])

with tab1:
    st.subheader("Ventas semanales")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weekly["Date"], y=weekly["Sales"],
            mode="lines", name="Ventas reales"
        )
    )
    fig.update_layout(
        xaxis_title="Fecha", yaxis_title="Ventas",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    if run:
        forecast = recursive_forecast(weekly, model, horizon)
        st.subheader(f"Pronóstico — próximas {horizon} semana(s)")

        table = forecast.copy()
        table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
        table["Predicted_Sales"] = table["Predicted_Sales"].map(
            lambda x: f"${x:,.2f}"
        )
        st.dataframe(table, use_container_width=True, hide_index=True)

        combined = go.Figure()
        combined.add_trace(
            go.Scatter(
                x=weekly["Date"], y=weekly["Sales"],
                mode="lines", name="Histórico"
            )
        )
        combined.add_trace(
            go.Scatter(
                x=forecast["Date"], y=forecast["Predicted_Sales"],
                mode="lines+markers", name="Pronóstico"
            )
        )
        combined.update_layout(
            xaxis_title="Fecha", yaxis_title="Ventas",
            hovermode="x unified"
        )
        st.plotly_chart(combined, use_container_width=True)

with tab2:
    st.subheader("Resultados sobre el bloque temporal de prueba")
    a, b, c = st.columns(3)
    a.metric("MAE", f"${metadata['random_forest_optimized']['MAE']:,.2f}")
    b.metric("RMSE", f"${metadata['random_forest_optimized']['RMSE']:,.2f}")
    c.metric("R²", f"{metadata['random_forest_optimized']['R2']:.4f}")

    st.write(
        f"Validación cruzada temporal (MAE): "
        f"**${metadata['random_forest_cv_mae']:,.2f}**"
    )

    imp = pd.DataFrame({
        "Variable": metadata["rf_features"],
        "Importancia": model.feature_importances_,
    }).sort_values("Importancia", ascending=False)

    st.subheader("Importancia de variables")
    st.bar_chart(imp.head(10).set_index("Variable"))

    st.subheader("Hiperparámetros")
    st.json(metadata["best_params"])

with tab3:
    st.subheader("Metodología")
    st.markdown(
        """
        **Preparación:** fechas → serie semanal → variables temporales → rezagos
        → medias móviles sin leakage.

        **Validación:** división cronológica 80/20 y `TimeSeriesSplit(n_splits=5)`.

        **Modelo final:** Random Forest optimizado mediante búsqueda de hiperparámetros.

        **Métricas:** MAE, RMSE y R².
        """
    )
    st.info(
        "El pronóstico es una estimación estadística basada en el histórico; "
        "no constituye una garantía de ventas futuras."
    )
