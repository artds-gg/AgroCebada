import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

ROOT = r"c:\Users\areth\Documents\FIRA\DATASET_RETO_AGRO_2026"
PRO_PATH = ROOT + r"\Conjunto_datos_PRO_AgroCebada.csv"
BASICO_PATH = ROOT + r"\Conjunto_datos_BASICO_AgroCebada2026.csv"
TARGET_PATH = ROOT + r"\ID_area_rendimiento_70_30_Reto_AgroCebada.csv"


def add_temporal_features(frame, prefix=""):
    grouped = frame.groupby("ID_POLIGONO")
    features = grouped.agg(
        ndvi_media=(f"{prefix}ndvi_promedio", "mean"),
        ndvi_maximo=(f"{prefix}ndvi_promedio", "max"),
        ndvi_minimo=(f"{prefix}ndvi_promedio", "min"),
        ndvi_std=(f"{prefix}ndvi_promedio", "std"),
        ndvi_ultimo=(f"{prefix}ndvi_promedio", lambda values: values.iloc[-1]),
        ndvi_primer=(f"{prefix}ndvi_promedio", lambda values: values.iloc[0]),
        ndvi_delta=(f"{prefix}ndvi_promedio", lambda values: values.iloc[-1] - values.iloc[0]),
        ndvi_trend=(
            f"{prefix}ndvi_promedio",
            lambda values: np.polyfit(np.arange(len(values)), values, 1)[0]
            if len(values) > 1 else 0
        ),
        evi_media=(f"{prefix}evi_promedio", "mean"),
        evi_maximo=(f"{prefix}evi_promedio", "max"),
        evi_minimo=(f"{prefix}evi_promedio", "min"),
        evi_ultimo=(f"{prefix}evi_promedio", lambda values: values.iloc[-1]),
        lai_media=(f"{prefix}lai_promedio", "mean"),
        lai_maximo=(f"{prefix}lai_promedio", "max"),
        lai_minimo=(f"{prefix}lai_promedio", "min"),
        lai_ultimo=(f"{prefix}lai_promedio", lambda values: values.iloc[-1]),
        msavi_media=(f"{prefix}msavi_promedio", "mean"),
        msavi_maximo=(f"{prefix}msavi_promedio", "max"),
        msavi_minimo=(f"{prefix}msavi_promedio", "min"),
        msavi_ultimo=(f"{prefix}msavi_promedio", lambda values: values.iloc[-1]),
        nubosidad_media=("porcentaje_nubosidad", "mean"),
        nubosidad_max=("porcentaje_nubosidad", "max"),
        observaciones=(f"{prefix}ndvi_promedio", "count"),
        fecha_inicio=("fecha_captura", "min"),
        fecha_fin=("fecha_captura", "max")
    ).reset_index()
    features["dias_observados"] = (
        features["fecha_fin"] - features["fecha_inicio"]
    ).dt.days + 1
    return features.drop(columns=["fecha_inicio", "fecha_fin"])


def add_sentinel_features(frame):
    selected_indices = [
        "ndvi_promedio", "evi_promedio", "lai_promedio", "fapar_promedio",
        "savi_promedio", "ndwi_promedio", "nddi_promedio"
    ]
    aggregations = {}
    for column in selected_indices:
        aggregations[f"{column}_media"] = (column, "mean")
        aggregations[f"{column}_maximo"] = (column, "max")
        aggregations[f"{column}_minimo"] = (column, "min")
    aggregations["sentinel_observaciones"] = ("ndvi_promedio", "count")
    aggregations["sentinel_fecha_inicio"] = ("fecha_captura", "min")
    aggregations["sentinel_fecha_fin"] = ("fecha_captura", "max")
    features = frame.groupby("ID_POLIGONO").agg(**aggregations).reset_index()
    features["sentinel_dias_observados"] = (
        features["sentinel_fecha_fin"] - features["sentinel_fecha_inicio"]
    ).dt.days + 1
    return features.drop(columns=["sentinel_fecha_inicio", "sentinel_fecha_fin"])


def add_landsat_features(frame):
    aggregations = {
        "landsat_vi6t_media": ("vi6t_promedio", "mean"),
        "landsat_vi6t_maximo": ("vi6t_promedio", "max"),
        "landsat_vi6t_minimo": ("vi6t_promedio", "min"),
        "landsat_vi6t_std": ("vi6t_promedio", "std"),
        "landsat_observaciones": ("vi6t_promedio", "count"),
        "landsat_fecha_inicio": ("fecha_captura", "min"),
        "landsat_fecha_fin": ("fecha_captura", "max")
    }
    features = frame.groupby("ID_POLIGONO").agg(**aggregations).reset_index()
    features["landsat_dias_observados"] = (
        features["landsat_fecha_fin"] - features["landsat_fecha_inicio"]
    ).dt.days + 1
    return features.drop(columns=["landsat_fecha_inicio", "landsat_fecha_fin"])


pro = pd.read_csv(PRO_PATH)
basico = pd.read_csv(BASICO_PATH)
target = pd.read_csv(TARGET_PATH)

for frame in [pro, basico]:
    frame["fecha_captura"] = pd.to_datetime(
        frame["fecha_captura"], dayfirst=True, errors="coerce"
    )

pro = pro.sort_values(["ID_POLIGONO", "fecha_captura"])
pro = pro[pro["porcentaje_nubosidad"] < 100].copy()
pro.loc[pro["lai_promedio"] < 0, "lai_promedio"] = np.nan
pro = pro[pro["fecha_captura"].dt.year == 2025].copy()

basico = basico[
    (basico["sensor"] == "Sentinel-2")
    & (basico["porcentaje_nubosidad"] < 100)
    & (basico["fecha_captura"].dt.year == 2025)
].copy()

landsat = pd.read_csv(BASICO_PATH)
landsat["fecha_captura"] = pd.to_datetime(
    landsat["fecha_captura"], dayfirst=True, errors="coerce"
)
landsat = landsat[
    (landsat["sensor"] == "Landsat")
    & (landsat["porcentaje_nubosidad"] < 100)
    & (landsat["fecha_captura"].dt.year == 2025)
].copy()

for frame in [pro, basico]:
    numeric_columns = frame.select_dtypes(include=np.number).columns
    frame[numeric_columns] = frame[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
numeric_columns = landsat.select_dtypes(include=np.number).columns
landsat[numeric_columns] = landsat[numeric_columns].apply(
    pd.to_numeric, errors="coerce"
)

pro_features = add_temporal_features(pro)
sentinel_features = add_sentinel_features(basico)
landsat_features = add_landsat_features(landsat)

parcel_features = pro_features.merge(
    sentinel_features,
    on="ID_POLIGONO",
    how="left"
).merge(
    landsat_features,
    on="ID_POLIGONO",
    how="left"
).merge(
    target[["ID_POLIGONO", "AREA_HA", "RENDIMIENTO_T_HA", "CONJUNTO"]],
    on="ID_POLIGONO",
    how="left"
)

training = parcel_features[
    parcel_features["CONJUNTO"] == "ENTRENAMIENTO"
].dropna(subset=["RENDIMIENTO_T_HA"]).copy()

pro_columns = [column for column in pro_features.columns if column != "ID_POLIGONO"]
sentinel_columns = [column for column in sentinel_features.columns if column != "ID_POLIGONO"]
landsat_columns = [column for column in landsat_features.columns if column != "ID_POLIGONO"]
pro_columns.append("AREA_HA")

feature_sets = {
    "pro_2025": pro_columns,
    "pro_sentinel2_2025": pro_columns + sentinel_columns,
    "pro_landsat_2025": pro_columns + landsat_columns
}

model = Pipeline([
    ("imputacion", SimpleImputer(strategy="median")),
    ("modelo", ExtraTreesRegressor(n_estimators=500, random_state=42))
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
for name, columns in feature_sets.items():
    X = training[columns]
    y = training["RENDIMIENTO_T_HA"]
    mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error"))
    results.append({
        "modelo": name,
        "variables": len(columns),
        "parcelas": len(training),
        "MAE_CV_media": mae.mean(),
        "RMSE_CV_media": rmse.mean()
    })

print(pd.DataFrame(results).round(3).to_string(index=False))

prediction = parcel_features[
    parcel_features["CONJUNTO"] == "PREDICCION"
].copy()
prediction_output = prediction[["ID_POLIGONO", "AREA_HA"]].copy()

for name, columns in feature_sets.items():
    model.fit(training[columns], training["RENDIMIENTO_T_HA"])
    prediction_output[f"rendimiento_predicho_{name}"] = model.predict(
        prediction[columns]
    )

prediction_output.to_csv(
    "predicciones_modelo_PRO_sensores.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nPredicciones generadas:", len(prediction_output))
print("Archivo guardado: predicciones_modelo_PRO_sensores.csv")
print(prediction_output.head(10).round(3).to_string(index=False))