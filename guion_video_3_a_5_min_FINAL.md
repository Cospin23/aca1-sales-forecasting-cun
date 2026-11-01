# Guion de sustentación — ACA 1 (3–5 minutos)

## 0:00–0:30 | Presentación
“Este proyecto corresponde al ACA 1 de Aprendizaje Automático de la
Especialización en Inteligencia Artificial de la CUN. El problema pertenece al
pilar de Ventas y utiliza el dataset Sales Forecasting - Retail Superstore.”

## 0:30–1:05 | Problema y objetivo
“El objetivo es utilizar el histórico de ventas para construir un modelo de
Machine Learning clásico capaz de estimar ventas futuras y presentar el
resultado mediante una aplicación interactiva.”

## 1:05–1:50 | Datos y preparación
“El dataset contiene 9.800 registros y 18 variables. Encontramos 11 valores
nulos en Postal Code y no encontramos duplicados exactos. Las fechas abarcan
de 2015 a 2018. Para forecasting transformamos las transacciones a una serie
semanal de 209 observaciones.”

“Después construimos rezagos de 1, 2, 4, 8, 13, 26 y 52 semanas y medias móviles
de 4, 8 y 13 semanas. Las medias móviles utilizan el desplazamiento de una semana
para evitar data leakage.”

## 1:50–2:35 | Modelos y evaluación
“Utilizamos una separación cronológica de 125 observaciones para entrenamiento
y 32 para prueba. El baseline obtuvo un MAE de $6.916,75.”

“La Regresión Lineal con variables cíclicas obtuvo MAE de $5.032,24, RMSE de
$6.285,68 y R² de 0,3106.”

“Para Random Forest realizamos una búsqueda de hiperparámetros con
TimeSeriesSplit. El mejor modelo utiliza 500 árboles, profundidad máxima 5,
max_features de 0,7 y min_samples_leaf de 1.”

“En el bloque de prueba obtuvo MAE de $5.349,13, RMSE de $6.675,79 y R² de
0,2224. Su MAE promedio en validación temporal fue $4.196,67.”

## 2:35–3:45 | Demostración de Streamlit
“Aquí podemos observar la aplicación. Presenta el histórico semanal, permite
seleccionar un horizonte de 1, 2, 4 u 8 semanas y genera el pronóstico de forma
recursiva.”

“También podemos consultar las métricas, la importancia de las variables y los
hiperparámetros utilizados.”

## 3:45–4:30 | Interpretación
“Los rezagos y medias móviles permiten al modelo utilizar información histórica
reciente y de diferentes horizontes. El resultado debe interpretarse como una
estimación estadística y no como una garantía de ventas futuras.”

## 4:30–5:00 | Cierre
“En conclusión, el proyecto integra preparación de datos, análisis exploratorio,
feature engineering, comparación de modelos clásicos, validación temporal y
despliegue mediante Streamlit, completando el ciclo de una solución de
aprendizaje automático aplicada a ventas.”
