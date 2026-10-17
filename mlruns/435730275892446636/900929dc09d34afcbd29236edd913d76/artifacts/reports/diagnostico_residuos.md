# Diagnóstico de residuos — Paso 5 (entrenamiento + MLflow)

- Modelo reentrenado: **HuberRegressor** (allowlist lineal).
- Conjunto de ajuste: `train_valid`.
- Smearing de Duan: `S = 1.010138`.
- Fecha (UTC): 2026-10-17T01:42:45+00:00.

## Escala log (homocedasticidad)
- Residuos log — media 0.0614, std 0.1290, skew 0.001, kurtosis 0.638 (valid).
- Q-Q: desviación máxima |z| = 4.26 (valid), 3.08 (test). Valores altos confirman las colas pesadas observadas en la etapa 04.
- Kurtosis > 0 implica colas más pesadas que una normal en escala log: el Q-Q se separa en los extremos (patrón previo: colas pesadas).

## Escala RUB (impacto de negocio)
- La retransformación a RUB reintroduce el abanico: Spearman(|residuo|, predicho) pasa de 0.178 (log) a 0.552 (RUB). Coherente con lo visto en la etapa 04: en log hay relativa homocedasticidad y en RUB la varianza crece con el nivel de precio.
- Cola alta (>= 200,000 RUB, valid): n=4 filas (0.22%), sesgo medio (real - predicho) = 37,643 RUB, MAPE = 16.45%.
- **Subestimación persistente de la cola alta**: con familia lineal las predicciones quedan por debajo del valor real en promedio 37,643 RUB. Esperado: la restricción lineal acota el rango predicho; el alivio viene de splines + smearing, no de más datos.
- Sesgo global (real - predicho) en valid: 3,196 RUB; MAPE valid = 10.30%.

## Comparación con las gráficas previas (etapa 04)
- Predicciones previas de valid (`predictions_valid.csv`): abanico Spearman = 0.965, sesgo cola alta = 236,888 RUB, MAPE = 99.97%.
- Deltas (05 − 04): abanico -0.413, sesgo cola -199,245 RUB. Un sesgo de cola menos positivo indica mejoría en la subestimación.

- Recuerde: el objetivo de modelado es `log1p(monthly_rent_rub)`; en RUB la heterocedasticidad reaparece por la retransformación. El diagnóstico decisivo de homocedasticidad es el de escala log; el de RUB mide impacto de negocio.
