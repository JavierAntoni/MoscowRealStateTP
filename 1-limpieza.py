import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuración de logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ExtremeCleaningPipeline")

# Coordenadas de referencia del centro de Moscú (Plaza Roja / Kremlin)
MOSCOW_CENTER_LAT = 55.7558
MOSCOW_CENTER_LON = 37.6173


def haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: float = MOSCOW_CENTER_LAT,
    lon2: float = MOSCOW_CENTER_LON,
) -> pd.Series:
    """Calcula la distancia geodésica Haversine en kilómetros."""
    r_earth = 6371.0
    phi1, lam1, phi2, lam2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * r_earth * np.arcsin(np.sqrt(a))


def load_and_standardize(filepath: str) -> pd.DataFrame:
    """
    1. Carga el dataset, limpia cadenas de texto, valida tipos y elimina duplicados.
    """
    logger.info(f"Cargando dataset desde: {filepath}")
    df = pd.read_csv(filepath)
    initial_rows = len(df)

    # Normalización de strings en columnas categóricas
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # Eliminación de duplicados exactos (ignorando 'id')
    feature_cols_no_id = [c for c in df.columns if c != "id"]
    df = df.drop_duplicates(subset=feature_cols_no_id, keep="first").copy()

    # Eliminación de duplicados físicos en la misma fecha
    physical_cols = [
        "date_posted", "lat", "lon", "total_area",
        "rooms", "floor", "total_floors", "building_year"
    ]
    df = df.drop_duplicates(subset=physical_cols, keep="first").copy()

    logger.info(
        f"Estandarización completada: {initial_rows} -> {len(df)} filas "
        f"({initial_rows - len(df)} duplicados eliminados)."
    )
    return df


def clean_temporal_and_physical_rules(
    df: pd.DataFrame,
    fix_historical_floors: bool = True,
) -> pd.DataFrame:
    """
    2. Limpia incoherencias temporales, físicas, arquitectónicas y geográficas.
    """
    df = df.copy()
    n_start = len(df)

    # Conversión y desglose temporal
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
    df = df.dropna(subset=["date_posted"]).copy()

    df["posted_year"] = df["date_posted"].dt.year
    df["posted_month"] = df["date_posted"].dt.month
    df["posted_quarter"] = df["date_posted"].dt.quarter

    # Regla 1: Eliminar inmuebles cuyo año de construcción sea posterior al año de publicación
    mask_future_building = df["building_year"] > df["posted_year"]
    n_future = mask_future_building.sum()
    df = df[~mask_future_building].copy()
    logger.info(f"Regla Temporal (building_year > posted_year): {n_future} filas eliminadas.")

    # Regla 2: Consistencia básica de pisos y área
    mask_valid_structure = (
        (df["floor"] >= 1)
        & (df["total_floors"] >= 1)
        & (df["floor"] <= df["total_floors"])
        & (df["total_area"] >= 15.0)
        & (df["rooms"] >= 1)
        & (df["monthly_rent_rub"] > 0)
    )
    n_invalid_struct = (~mask_valid_structure).sum()
    df = df[mask_valid_structure].copy()
    logger.info(f"Regla Estructural básica: {n_invalid_struct} filas eliminadas.")

    # Regla 3: Corrección de incoherencias arquitectónicas (Khrushchev y Stalin)
    # En este dataset, total_floors fue muestreado aleatoriamente en [5, 9, 12, 17, 22].
    # Ajustamos topes arquitectónicos realistas sin destruir el 25% de la muestra.
    df["arch_floor_anomaly"] = (
        ((df["building_type"] == "khrushchev") & (df["total_floors"] > 5))
        | ((df["building_type"] == "stalin") & (df["total_floors"] > 12))
    ).astype(int)

    if fix_historical_floors:
        khrushchev_mask = (df["building_type"] == "khrushchev") & (df["total_floors"] > 5)
        df.loc[khrushchev_mask, "total_floors"] = 5
        df.loc[khrushchev_mask & (df["floor"] > 5), "floor"] = 5

        stalin_mask = (df["building_type"] == "stalin") & (df["total_floors"] > 12)
        df.loc[stalin_mask, "total_floors"] = 9
        df.loc[stalin_mask & (df["floor"] > 9), "floor"] = 9
        logger.info(
            f"Corrección arquitectónica aplicada en {df['arch_floor_anomaly'].sum()} registros "
            "(Khrushchev acotado a 5 pisos, Stalin a 9-12 pisos)."
        )

    # Regla 4: Verificación geodésica de coordenadas vs to_center_km
    df["verified_center_km"] = haversine_km(df["lat"], df["lon"]).round(2)
    coord_error = (df["to_center_km"] - df["verified_center_km"]).abs()
    mask_bad_coords = coord_error > 1.5  # Tolerancia de 1.5 km
    if mask_bad_coords.sum() > 0:
        logger.info(f"Coordenadas inconsistentes eliminadas: {mask_bad_coords.sum()}")
        df = df[~mask_bad_coords].copy()

    # Regla 5: Corrección de metro_distance_min == 0 (imposible 0 min exactos puerta a andén)
    n_zero_metro = (df["metro_distance_min"] <= 0).sum()
    if n_zero_metro > 0:
        df.loc[df["metro_distance_min"] <= 0, "metro_distance_min"] = 1
        logger.info(f"Ajustados {n_zero_metro} registros con metro_distance_min <= 0 a 1 min.")

    logger.info(f"Total tras reglas físicas y temporales: {n_start} -> {len(df)} filas.")
    return df


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    3. Ingeniería de características de dominio inmobiliario para EDA y MLflow.
    """
    df = df.copy()

    # Antigüedad del edificio al momento de la publicación
    df["building_age_at_post"] = df["posted_year"] - df["building_year"]

    # Métricas de espacio y estructura
    df["area_per_room"] = (df["total_area"] / df["rooms"]).round(2)
    df["floor_ratio"] = (df["floor"] / df["total_floors"]).round(4)
    df["is_first_floor"] = (df["floor"] == 1).astype(int)
    df["is_last_floor"] = (df["floor"] == df["total_floors"]).astype(int)

    # Codificación cíclica del mes de publicación (captura estacionalidad sin saltos entre dic-ene)
    df["posted_month_sin"] = np.sin(2 * np.pi * df["posted_month"] / 12.0).round(6)
    df["posted_month_cos"] = np.cos(2 * np.pi * df["posted_month"] / 12.0).round(6)

    # Codificación ordinal con sentido de negocio para 'renovation'
    renovation_order = {
        "no_renovation": 0,
        "cosmetic": 1,
        "euro": 2,
        "designer": 3,
    }
    df["renovation_ordinal"] = df["renovation"].map(renovation_order)

    # Indicador de anomalía cuando la estación de metro asignada está a más de 45 min a pie
    df["is_far_metro_outlier"] = (df["metro_distance_min"] > 45).astype(int)

    # Transformación logarítmica de distancia al metro y área (alta linealidad con log-renta)
    df["log_total_area"] = np.log(df["total_area"]).round(6)
    df["log_metro_dist"] = np.log1p(df["metro_distance_min"]).round(6)

    # Recálculo exacto de rent_per_sqm SOLO para el dataset de análisis (EDA)
    df["rent_per_sqm"] = (df["monthly_rent_rub"] / df["total_area"]).round(2)

    return df


def remove_statistical_and_multivariate_outliers(
    df: pd.DataFrame,
    z_resid_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    4. Filtra outliers univariados por grupo (area_per_room) y outliers multivariados
    respecto al comportamiento estructural del target (Residuos Studentizados y Distancia de Cook).
    """
    df = df.copy()
    n_before = len(df)

    # A. Filtro IQR extremo (3.0 * IQR) en area_per_room agrupado por número de habitaciones
    keep_iqr = pd.Series(True, index=df.index)
    for r, grp in df.groupby("rooms"):
        q1 = grp["area_per_room"].quantile(0.25)
        q3 = grp["area_per_room"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        keep_iqr.loc[grp.index] = grp["area_per_room"].between(lower, upper)

    n_iqr_removed = (~keep_iqr).sum()
    df = df[keep_iqr].copy()
    logger.info(f"Filtro IQR por habitaciones (area_per_room): {n_iqr_removed} filas eliminadas.")

    # B. Filtro Multivariado: Modelo OLS de diagnóstico sobre log(monthly_rent_rub)
    df["_log_rent_tmp"] = np.log(df["monthly_rent_rub"])
    formula = (
        "_log_rent_tmp ~ log_total_area + C(rooms) + C(building_type) + "
        "C(renovation) + furnished + pets_allowed + metro_distance_min + "
        "verified_center_km + C(district) + posted_year"
    )
    diag_model = smf.ols(formula=formula, data=df).fit()
    influence = diag_model.get_influence()

    # Residuos estandarizados internamente y Distancia de Cook
    std_resid = influence.resid_studentized_internal
    cooks_d = influence.cooks_distance[0]
    cooks_cutoff = 4.0 / len(df)

    # Eliminamos puntos con residuo > 3 sigma o (residuo > 2.5 sigma Y alta influencia de Cook)
    mask_multivariate_outlier = (np.abs(std_resid) > z_resid_threshold) | (
        (np.abs(std_resid) > 2.5) & (cooks_d > cooks_cutoff * 3.0)
    )
    n_multi_removed = mask_multivariate_outlier.sum()

    df = df[~mask_multivariate_outlier].drop(columns=["_log_rent_tmp"]).copy()
    logger.info(
        f"Filtro Multivariado (OLS Residuos > {z_resid_threshold}σ / Cook's D): "
        f"{n_multi_removed} filas eliminadas (R² diagnóstico = {diag_model.rsquared:.4f})."
    )
    logger.info(f"Filas restantes tras filtrado estadístico: {n_before} -> {len(df)}.")
    return df


def build_eda_and_mlflow_datasets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    5. Separa el dataset limpio en dos versiones:
       - df_eda: Completo e interpretable para Análisis Exploratorio.
       - df_mlflow: Sin Data Leakage, sin identificadores y con tipos float64/string
                    listos para mlflow.models.infer_signature() y mlflow.data.from_pandas().
    """
    # 1. Dataset para Análisis Exploratorio (EDA)
    df_eda = df.copy()
    df_eda["date_posted"] = df_eda["date_posted"].dt.strftime("%Y-%m-%d")

    # 2. Dataset para Entrenamiento en MLflow (prevención estricta de Data Leakage)
    leakage_and_id_cols = [
        "id",                # Identificador único sin valor generalizable
        "date_posted",       # Fecha cruda (ya extraída en año/mes/trimestre/ciclos)
        "rent_per_sqm",      # TARGET LEAKAGE DIRECTO (monthly_rent_rub / total_area)
        "to_center_km",      # Redundante con verified_center_km calculado con Haversine
    ]
    df_mlflow = df.drop(columns=leakage_and_id_cols, errors="ignore").copy()

    # Conversión de booleanos y enteros a float64 para compatibilidad total con MLflow Schemas
    # (MLflow exige float64 en columnas numéricas para soportar posibles NaNs en inferencia)
    bool_cols = df_mlflow.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df_mlflow[col] = df_mlflow[col].astype("float64")

    num_cols = df_mlflow.select_dtypes(include=["int64", "int32", "float32", "float64"]).columns
    for col in num_cols:
        df_mlflow[col] = df_mlflow[col].astype("float64")

    # Reordenar para que el target 'monthly_rent_rub' quede al final
    target_col = "monthly_rent_rub"
    feature_cols = [c for c in df_mlflow.columns if c != target_col]
    df_mlflow = df_mlflow[feature_cols + [target_col]]

    return df_eda, df_mlflow


def run_cleaning_pipeline(
    input_csv: str = "data/rentals.csv",
    output_eda_csv: str = "data/rentals_clean_analysis.csv",
    output_mlflow_csv: str = "data/rentals_clean_mlflow.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el pipeline completo de extremo a extremo y guarda los artefactos."""
    df_raw = load_and_standardize(input_csv)
    df_phys = clean_temporal_and_physical_rules(df_raw, fix_historical_floors=True)
    df_feat = engineer_domain_features(df_phys)
    df_clean = remove_statistical_and_multivariate_outliers(df_feat, z_resid_threshold=3.0)

    df_eda, df_mlflow = build_eda_and_mlflow_datasets(df_clean)

    df_eda.to_csv(output_eda_csv, index=False)
    df_mlflow.to_csv(output_mlflow_csv, index=False)

    logger.info("=" * 65)
    logger.info("RESUMEN FINAL DE LIMPIEZA EXTREMA")
    logger.info(f"Registros originales         : {len(df_raw):,}")
    logger.info(f"Registros finales limpios    : {len(df_clean):,} ({len(df_clean)/len(df_raw)*100:.2f}%)")
    logger.info(f"Registros descartados        : {len(df_raw) - len(df_clean):,}")
    logger.info(f"Archivo EDA guardado en      : {output_eda_csv} ({df_eda.shape[1]} columnas)")
    logger.info(f"Archivo MLflow guardado en   : {output_mlflow_csv} ({df_mlflow.shape[1]} columnas)")
    logger.info("=" * 65)

    return df_eda, df_mlflow


if __name__ == "__main__":
    run_cleaning_pipeline()