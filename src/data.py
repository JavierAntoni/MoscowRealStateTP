import pandas as pd

# 1. Cargar datos
df = pd.read_csv("rentals.csv")

# 2. Auditoría general (Paso 2 de la guía)
print(f"Dimensiones: {df.shape}")
print("\nTipos de datos y nulos:")
print(df.info())
print("\nDuplicados:", df.duplicated().sum())

# 3. Estadísticas descriptivas (numéricas y categóricas)
display(df.describe().T)
display(df.describe(include=["object", "bool"]).T)

# 4. Verificación rápida de consistencia lógica
print("Pisos mayores al total del edificio:", (df["floor"] > df["total_floors"]).sum())

# 5. Separar objetivo y descartar columnas prohibidas (Paso 3)
objetivo = "monthly_rent_rub"
columnas_excluir = ["id", "rent_per_sqm"]  # Excluir id y variable hermana del objetivo
df_trabajo = df.drop(columns=columnas_excluir)