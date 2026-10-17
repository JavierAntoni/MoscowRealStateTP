# Fórmula Ganadora y Coeficientes — Modelo `HuberRegressor`

- **Familia del modelo:** `HuberRegressor`
- **Intercepto (b0):** `0.108304`
- **Factor de Duan Smearing (S):** `1.010138`
- **Variables totales tras preprocesamiento:** `30` (activas != 0: `30`)

## 1. Ecuación en escala logarítmica
```text
y_log_hat = 0.108304 + 0.991630 * (target_enc__district) + 0.359726 * (numeric__total_area) + 0.251684 * (onehot__renovation_designer) - 0.169272 * (onehot__building_type_khrushchev) + 0.164307 * (onehot__building_type_stalin) - 0.163866 * (onehot__renovation_no_renovation) + 0.153095 * (spline__to_center_km_sp_0) + 0.118209 * (onehot__building_type_monolith) + 0.082333 * (onehot__renovation_euro) + 0.079749 * (numeric__furnished) - 0.076371 * (onehot__building_type_panel) - 0.061930 * (onehot__renovation_cosmetic) - 0.061523 * (numeric__rooms) - 0.055105 * (spline__to_center_km_sp_1) + 0.049511 * (onehot__building_type_brick) - 0.046639 * (spline__to_center_km_sp_5) - 0.039078 * (spline__to_center_km_sp_3) - 0.037301 * (numeric__area_per_room) - 0.026723 * (numeric__pets_allowed) + 0.022897 * (numeric__building_age) - 0.022866 * (spline__to_center_km_sp_4) + 0.021836 * (onehot__building_type_modern_panel) - 0.014855 * (spline__to_center_km_sp_2) + 0.008848 * (spline__to_center_km_sp_6) + 0.003567 * (numeric__lat) - 0.002440 * (numeric__lon) - 0.001954 * (numeric__floor_ratio) + 0.001276 * (numeric__floor) - 0.001175 * (numeric__total_floors) + 0.001119 * (numeric__deposit_months)
```

## 2. Ecuación retransformada a Rublos (con Duan Smearing)
```text
y_rub_hat = exp(y_log_hat) * 1.010138 - 1
```

## 3. Top 25 Variables con mayor peso (Variables estandarizadas)
| Ranking | Variable | Coeficiente | Impacto aprox. en RUB por +1 unidad (%) |
|---:|---|---:|---:|
| 1 | `target_enc__district` | +0.991630 | +169.56% |
| 2 | `numeric__total_area` | +0.359726 | +43.29% |
| 3 | `onehot__renovation_designer` | +0.251684 | +28.62% |
| 4 | `onehot__building_type_khrushchev` | -0.169272 | -15.57% |
| 5 | `onehot__building_type_stalin` | +0.164307 | +17.86% |
| 6 | `onehot__renovation_no_renovation` | -0.163866 | -15.11% |
| 7 | `spline__to_center_km_sp_0` | +0.153095 | +16.54% |
| 8 | `onehot__building_type_monolith` | +0.118209 | +12.55% |
| 9 | `onehot__renovation_euro` | +0.082333 | +8.58% |
| 10 | `numeric__furnished` | +0.079749 | +8.30% |
| 11 | `onehot__building_type_panel` | -0.076371 | -7.35% |
| 12 | `onehot__renovation_cosmetic` | -0.061930 | -6.01% |
| 13 | `numeric__rooms` | -0.061523 | -5.97% |
| 14 | `spline__to_center_km_sp_1` | -0.055105 | -5.36% |
| 15 | `onehot__building_type_brick` | +0.049511 | +5.08% |
| 16 | `spline__to_center_km_sp_5` | -0.046639 | -4.56% |
| 17 | `spline__to_center_km_sp_3` | -0.039078 | -3.83% |
| 18 | `numeric__area_per_room` | -0.037301 | -3.66% |
| 19 | `numeric__pets_allowed` | -0.026723 | -2.64% |
| 20 | `numeric__building_age` | +0.022897 | +2.32% |
| 21 | `spline__to_center_km_sp_4` | -0.022866 | -2.26% |
| 22 | `onehot__building_type_modern_panel` | +0.021836 | +2.21% |
| 23 | `spline__to_center_km_sp_2` | -0.014855 | -1.47% |
| 24 | `spline__to_center_km_sp_6` | +0.008848 | +0.89% |
| 25 | `numeric__lat` | +0.003567 | +0.36% |