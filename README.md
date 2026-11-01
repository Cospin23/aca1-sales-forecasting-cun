# ACA 1 — Aprendizaje Automático | CUN

## Entrega final — Sales Forecasting

### Entregables oficiales
1. `notebooks/ACA1_Sales_Forecasting_CUN.ipynb`
2. `poster_ACA1_Sales_Forecasting_CUN_FINAL.pdf`
3. `app.py` + `requirements.txt` + `src/` + `models/`
4. `guion_video_3_a_5_min.md`

### Archivos de soporte
- `informe_tecnico_ACA1_CUN.pdf`
- `train_model.py`
- `models/metadata.json`
- `models/sales_forecasting_rf.pkl`
- `data/train.csv`

### Ejecución
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

### Modelo de despliegue
Random Forest Regressor optimizado con TimeSeriesSplit.

**Hiperparámetros:** 500 árboles, profundidad máxima 5, `max_features=0.7`,
`min_samples_leaf=1`.

**Resultados test:** MAE $5.349,13 · RMSE $6.675,79 · R² 0,2224.

**Validación temporal:** MAE CV $4.196,67.

La Regresión Lineal con variables cíclicas obtuvo MAE $5.032,24 en el bloque
test; Random Forest se seleccionó para despliegue por su desempeño promedio en
la validación temporal.
