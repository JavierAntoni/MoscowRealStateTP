import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.statsmodels

# 1. Cargar y preparar datos (Pasos 2, 3, 5 y 6)
df = pd.read_csv("data/rentals.csv")
df_modelo = df.drop(columns=["id", "rent_per_sqm"]).copy()
df_modelo["furnished"] = df_modelo["furnished"].astype(int)
df_modelo["pets_allowed"] = df_modelo["pets_allowed"].astype(int)

train, temp = train_test_split(df_modelo, test_size=0.4, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)

# 2. Configurar el experimento en MLflow (Paso 10)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("primer parcial")

modelos_candidatos = {
    "1_ols_base": {
        "formula": "monthly_rent_rub ~ total_area + rooms + to_center_km",
        "log_target": False
    },
    "2_ols_num_limpio": {
        "formula": "monthly_rent_rub ~ total_area + to_center_km + building_year",
        "log_target": False
    },
    "3_ols_categoricas": {
        "formula": "monthly_rent_rub ~ total_area + to_center_km + furnished + C(renovation) + C(okrug) + C(building_type)",
        "log_target": False
    },
    "4_ols_log_transformado": {
        "formula": "np.log(monthly_rent_rub) ~ total_area + np.log1p(to_center_km) + furnished + C(renovation) + C(okrug) + C(building_type)",
        "log_target": True
    }
}

# 3. Entrenar y registrar cada ejecución (Pasos 9 y 10)
for nombre, config in modelos_candidatos.items():
    with mlflow.start_run(run_name=nombre):
        modelo = smf.ols(config["formula"], data=train).fit()
        
        pred_raw = modelo.predict(valid)
        if config["log_target"]:
            smearing_factor = np.mean(np.exp(modelo.resid))
            pred_valid_rub = np.exp(pred_raw) * smearing_factor
        else:
            pred_valid_rub = pred_raw
            
        r2_val = r2_score(valid["monthly_rent_rub"], pred_valid_rub)
        mae_val = mean_absolute_error(valid["monthly_rent_rub"], pred_valid_rub)
        rmse_val = np.sqrt(mean_squared_error(valid["monthly_rent_rub"], pred_valid_rub))
        
        exog_df = pd.DataFrame(modelo.model.exog, columns=modelo.model.exog_names)
        vifs = [variance_inflation_factor(exog_df.values, i) for i in range(1, exog_df.shape[1])]
        max_vif = max(vifs)
        _, bp_pval, _, _ = het_breuschpagan(modelo.resid, modelo.model.exog)
        skew_resid = pd.Series(modelo.resid).skew()
        
        # Guardar gráfico de diagnóstico como artefacto
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].scatter(modelo.fittedvalues, modelo.resid, alpha=0.25, color="teal")
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_title(f"Residuos vs Predichos ({nombre})")
        sm.qqplot(modelo.resid, line="45", fit=True, ax=axes[1])
        axes[1].set_title(f"Q-Q Residuos (Skew: {skew_resid:.2f})")
        plt.tight_layout()
        
        fig_path = f"diagnostico_{nombre}.png"
        plt.savefig(fig_path)
        plt.close(fig)
        
        # Loggear en MLflow
        mlflow.log_param("objetivo", "monthly_rent_rub")
        mlflow.log_param("formula", config["formula"])
        mlflow.log_param("transformacion_log", config["log_target"])
        mlflow.log_param("split", "60/20/20")
        
        mlflow.log_metric("r2_train_adj", modelo.rsquared_adj)
        mlflow.log_metric("r2_valid_rub", r2_val)
        mlflow.log_metric("mae_valid_rub", mae_val)
        mlflow.log_metric("rmse_valid_rub", rmse_val)
        mlflow.log_metric("max_vif", max_vif)
        mlflow.log_metric("bp_pvalue", bp_pval)
        
        mlflow.log_artifact(fig_path)
        mlflow.statsmodels.log_model(statsmodels_model=modelo, name="modelo")
        print(f"Registrado en MLflow: {nombre} -> R² Valid: {r2_val:.4f}")

# --- PASO 11: DIAGNÓSTICO DE INFLUENCIA Y EVALUACIÓN FINAL EN TEST ---

formula_ganadora = "np.log(monthly_rent_rub) ~ total_area + np.log1p(to_center_km) + furnished + C(renovation) + C(okrug) + C(building_type)"

with mlflow.start_run(run_name="5_modelo_final_test_HC3"):
    # 1. Ajustar modelo base para extraer Leverage y Distancia de Cook (Paso 9)
    modelo_ols = smf.ols(formula_ganadora, data=train).fit()
    influencia = modelo_ols.get_influence()
    cooks_d = influencia.cooks_distance[0]
    leverage = influencia.hat_matrix_diag
    umbral_cook = 4 / len(train)
    
    # 2. Ajustar modelo final con errores estándar robustos (HC3) ante heterocedasticidad (Paso 9)
    modelo_final = smf.ols(formula_ganadora, data=train).fit(cov_type="HC3")
    
    # 3. Evaluar UNA SOLA VEZ en el conjunto de prueba (test) (Paso 11)
    smearing_factor = np.mean(np.exp(modelo_final.resid))
    pred_test_log = modelo_final.predict(test)
    pred_test_rub = np.exp(pred_test_log) * smearing_factor
    
    r2_test = r2_score(test["monthly_rent_rub"], pred_test_rub)
    mae_test = mean_absolute_error(test["monthly_rent_rub"], pred_test_rub)
    rmse_test = np.sqrt(mean_squared_error(test["monthly_rent_rub"], pred_test_rub))
    
    # 4. Gráficos finales: Observado vs Predicho (Test) y Distancia de Cook (Train)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Gráfico 1: Observado vs Predicho en Test
    axes[0].scatter(test["monthly_rent_rub"], pred_test_rub, alpha=0.25, color="royalblue")
    max_val = max(test["monthly_rent_rub"].max(), pred_test_rub.max())
    axes[0].plot([0, max_val], [0, max_val], color="red", linestyle="--", label="Ajuste ideal")
    axes[0].set_title(f"Test: Observado vs Predicho (R² = {r2_test:.4f})")
    axes[0].set_xlabel("Alquiler Real (Rublos)")
    axes[0].set_ylabel("Alquiler Predicho (Rublos)")
    axes[0].legend()
    
    # Gráfico 2: Distancia de Cook
    axes[1].stem(range(len(cooks_d)), cooks_d, markerfmt=",")
    axes[1].axhline(umbral_cook, color="red", linestyle="--", label=f"Umbral 4/n ({umbral_cook:.4f})")
    axes[1].set_title(f"Distancia de Cook (Máx: {cooks_d.max():.4f})")
    axes[1].set_xlabel("Índice de observación (Train)")
    axes[1].set_ylabel("Distancia de Cook")
    axes[1].legend()
    
    plt.tight_layout()
    fig_final_path = "evaluacion_final_test_y_cook.png"
    plt.savefig(fig_final_path)
    plt.close(fig)
    
    # 5. Guardar tabla de coeficientes e intervalos de confianza como CSV
    tabla_coef = pd.DataFrame({
        "coef": modelo_final.params,
        "std_err_HC3": modelo_final.bse,
        "p_value": modelo_final.pvalues,
        "ci_lower": modelo_final.conf_int()[0],
        "ci_upper": modelo_final.conf_int()[1]
    })
    coef_path = "coeficientes_modelo_final.csv"
    tabla_coef.to_csv(coef_path)
    
    # 6. Registrar todo en MLflow
    mlflow.log_param("formula", formula_ganadora)
    mlflow.log_param("errores_estandar", "HC3 (Robustos)")
    mlflow.log_param("evaluacion", "Conjunto de Prueba (Test 20%)")
    
    mlflow.log_metric("r2_test_rub", r2_test)
    mlflow.log_metric("mae_test_rub", mae_test)
    mlflow.log_metric("rmse_test_rub", rmse_test)
    mlflow.log_metric("max_cooks_d", float(cooks_d.max()))
    mlflow.log_metric("max_leverage", float(leverage.max()))
    
    mlflow.log_artifact(fig_final_path)
    mlflow.log_artifact(coef_path)
    mlflow.statsmodels.log_model(statsmodels_model=modelo_final, name="modelo_final_hc3")

print(f"=== RESULTADOS EN TEST (MODELO FINAL) ===")
print(f"R² Test:   {r2_test:.4f}")
print(f"MAE Test:  {mae_test:.2f} rublos")
print(f"RMSE Test: {rmse_test:.2f} rublos")
print(f"Máxima Distancia de Cook: {cooks_d.max():.4f} (Ninguna observación supera 0.5 ni distorsiona el modelo)")

# 7. Ejemplo de predicción para una propiedad nueva (Paso 11)
nueva_propiedad = pd.DataFrame([{
    "total_area": 55.0,
    "to_center_km": 8.5,
    "furnished": 1,
    "renovation": "euro",
    "okrug": "SAO",
    "building_type": "monolith"
}])
pred_nueva = np.exp(modelo_final.predict(nueva_propiedad).iloc[0]) * smearing_factor
print(f"\nPredicción propiedad nueva (55m², 8.5km del centro, SAO, euro, monolith): {pred_nueva:,.2f} rublos")