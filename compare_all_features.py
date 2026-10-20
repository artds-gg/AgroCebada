from pathlib import Path
from statistics import mean

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline

ROOT = Path(r"c:\Users\areth\Documents\FIRA\DATASET_RETO_AGRO_2026")
PRO_PATH = ROOT / "Conjunto_datos_PRO_AgroCebada.csv"
BASICO_PATH = ROOT / "Conjunto_datos_BASICO_AgroCebada2026.csv"
TARGET_PATH = ROOT / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv"
PARCELS_PATH = ROOT / "Parcelas_Reto_AGC_CONJUNTO_70_30" / "Parcelas_Reto_AGC_CONJUNTO.shp"


def clean_dates(frame):
    frame["fecha_captura"] = pd.to_datetime(
        frame["fecha_captura"], dayfirst=True, errors="coerce"
    )
    return frame


def temporal_features(frame, prefix=""):
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


def sensor_features(frame, sensor_name):
    selected_indices = [
        "ndvi_promedio", "evi_promedio", "lai_promedio", "fapar_promedio",
        "savi_promedio", "ndwi_promedio", "nddi_promedio"
    ]
    aggregations = {}
    for column in selected_indices:
        aggregations[f"{sensor_name}_{column}_media"] = (column, "mean")
        aggregations[f"{sensor_name}_{column}_maximo"] = (column, "max")
        aggregations[f"{sensor_name}_{column}_minimo"] = (column, "min")
    aggregations[f"{sensor_name}_observaciones"] = ("ndvi_promedio", "count")
    aggregations[f"{sensor_name}_fecha_inicio"] = ("fecha_captura", "min")
    aggregations[f"{sensor_name}_fecha_fin"] = ("fecha_captura", "max")
    features = frame.groupby("ID_POLIGONO").agg(**aggregations).reset_index()
    features[f"{sensor_name}_dias_observados"] = (
        features[f"{sensor_name}_fecha_fin"]
        - features[f"{sensor_name}_fecha_inicio"]
    ).dt.days + 1
    return features.drop(columns=[
        f"{sensor_name}_fecha_inicio", f"{sensor_name}_fecha_fin"
    ])


def landsat_features(frame):
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


def zonal_mean(parcels, raster_paths, prefix):
    output = parcels[["ID_POLIGONO"]].copy()
    for raster_path in sorted(raster_paths):
        with rasterio.open(raster_path) as source:
            projected = parcels.to_crs(source.crs)
            values = []
            for geometry in projected.geometry:
                try:
                    data, _ = mask(
                        source,
                        [mapping(geometry)],
                        crop=True,
                        filled=False,
                        all_touched=True
                    )
                    values.append(float(data[0].mean()) if data[0].count() else np.nan)
                except ValueError:
                    values.append(np.nan)
            output[f"{prefix}_{Path(raster_path).stem.lower()}"] = values
    return output


def summarize_raster_block(frame, prefix):
    columns = [column for column in frame.columns if column.startswith(prefix)]
    result = frame[["ID_POLIGONO"]].copy()
    result[f"{prefix}_media"] = frame[columns].mean(axis=1)
    result[f"{prefix}_std"] = frame[columns].std(axis=1)
    result[f"{prefix}_minimo"] = frame[columns].min(axis=1)
    result[f"{prefix}_maximo"] = frame[columns].max(axis=1)
    result[f"{prefix}_total"] = frame[columns].sum(axis=1)
    return result


def build_spatial_features(parcels):
    precipitation_paths = sorted(ROOT.rglob("PREC_2025_*.tif"))
    tmax_paths = sorted(ROOT.rglob("Tmax_2025_*.tif"))
    tmin_paths = sorted(ROOT.rglob("Tmin_2025_*.tif"))
    elevation_path = next(ROOT.rglob("Elevacion_INEGI_CEM4_120m.tif"))
    slope_path = next(ROOT.rglob("Pendiente_INEGI_CEM4_120m_grados.tif"))

    precipitation = zonal_mean(parcels, precipitation_paths, "prec")
    tmax = zonal_mean(parcels, tmax_paths, "tmax")
    tmin = zonal_mean(parcels, tmin_paths, "tmin")
    elevation = zonal_mean(parcels, [elevation_path], "elevacion")
    slope = zonal_mean(parcels, [slope_path], "pendiente")

    climate = precipitation.merge(tmax, on="ID_POLIGONO")
    climate = climate.merge(tmin, on="ID_POLIGONO")
    climate = climate.merge(summarize_raster_block(precipitation, "prec"), on="ID_POLIGONO")
    climate = climate.merge(summarize_raster_block(tmax, "tmax"), on="ID_POLIGONO")
    climate = climate.merge(summarize_raster_block(tmin, "tmin"), on="ID_POLIGONO")
    climate["temperatura_media_anual"] = (
        climate["tmax_media"] + climate["tmin_media"]
    ) / 2
    climate["amplitud_termica_media"] = (
        climate["tmax_media"] - climate["tmin_media"]
    )

    spatial = climate.merge(elevation, on="ID_POLIGONO")
    spatial = spatial.merge(slope, on="ID_POLIGONO")
    return spatial


pro = clean_dates(pd.read_csv(PRO_PATH))
basico = clean_dates(pd.read_csv(BASICO_PATH))
target = pd.read_csv(TARGET_PATH)

pro = pro[
    (pro["porcentaje_nubosidad"] < 100)
    & (pro["fecha_captura"].dt.year == 2025)
].sort_values(["ID_POLIGONO", "fecha_captura"])
pro.loc[pro["lai_promedio"] < 0, "lai_promedio"] = np.nan

sentinel = basico[
    (basico["sensor"] == "Sentinel-2")
    & (basico["porcentaje_nubosidad"] < 100)
    & (basico["fecha_captura"].dt.year == 2025)
].sort_values(["ID_POLIGONO", "fecha_captura"])
landsat = basico[
    (basico["sensor"] == "Landsat")
    & (basico["porcentaje_nubosidad"] < 100)
    & (basico["fecha_captura"].dt.year == 2025)
].sort_values(["ID_POLIGONO", "fecha_captura"])

for frame in [pro, sentinel, landsat]:
    numeric_columns = frame.select_dtypes(include=np.number).columns
    frame[numeric_columns] = frame[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

pro_features = temporal_features(pro)
sentinel_features = sensor_features(sentinel, "sentinel")
landsat_features_frame = landsat_features(landsat)

parcels = gpd.read_file(PARCELS_PATH).rename(columns={"ID_POLIGON": "ID_POLIGONO"})
parcels = parcels[["ID_POLIGONO", "geometry"]]
spatial_features = build_spatial_features(parcels)
spatial_features.to_csv(
    "variables_espaciales_2025.csv",
    index=False,
    encoding="utf-8-sig"
)
print("Valores espaciales faltantes:", int(spatial_features.isna().sum().sum()))

all_features = pro_features.merge(sentinel_features, on="ID_POLIGONO", how="left")
all_features = all_features.merge(landsat_features_frame, on="ID_POLIGONO", how="left")
all_features = all_features.merge(spatial_features, on="ID_POLIGONO", how="left")
all_features = all_features.merge(
    target[["ID_POLIGONO", "AREA_HA", "RENDIMIENTO_T_HA", "CONJUNTO"]],
    on="ID_POLIGONO",
    how="left"
)
all_features.to_csv(
    "features_modelo_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

training = all_features[
    all_features["CONJUNTO"] == "ENTRENAMIENTO"
].dropna(subset=["RENDIMIENTO_T_HA"]).copy()
prediction = all_features[
    all_features["CONJUNTO"] == "PREDICCION"
].copy()

pro_columns = [column for column in pro_features.columns if column != "ID_POLIGONO"]
sentinel_columns = [column for column in sentinel_features.columns if column != "ID_POLIGONO"]
landsat_columns = [column for column in landsat_features_frame.columns if column != "ID_POLIGONO"]
spatial_columns = [column for column in spatial_features.columns if column != "ID_POLIGONO"]
pro_columns.append("AREA_HA")

feature_sets = {
    "pro": pro_columns,
    "pro_landsat": pro_columns + landsat_columns,
    "pro_landsat_sentinel": pro_columns + landsat_columns + sentinel_columns,
    "pro_landsat_sentinel_clima": pro_columns + landsat_columns + sentinel_columns + [
        column for column in spatial_columns
        if column.startswith(("prec_", "tmax_", "tmin_", "temperatura_", "amplitud_"))
    ],
    "pro_landsat_sentinel_clima_topografia": pro_columns + landsat_columns + sentinel_columns + spatial_columns
}

model = Pipeline([
    ("imputacion", SimpleImputer(strategy="median")),
    ("modelo", ExtraTreesRegressor(n_estimators=400, random_state=42, min_samples_leaf=2))
])
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
results = []
for name, columns in feature_sets.items():
    X = training[columns]
    y = training["RENDIMIENTO_T_HA"]
    mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1))
    results.append({
        "modelo": name,
        "variables": len(columns),
        "parcelas": len(training),
        "MAE_CV_media": mae.mean(),
        "MAE_CV_std": mae.std(),
        "RMSE_CV_media": rmse.mean(),
        "RMSE_CV_std": rmse.std()
    })

results_frame = pd.DataFrame(results).sort_values("MAE_CV_media")
print(results_frame.round(3).to_string(index=False))
results_frame.to_csv("comparacion_features_espaciales.csv", index=False, encoding="utf-8-sig")

best_name = results_frame.iloc[0]["modelo"]
best_columns = feature_sets[best_name]
Path("features_seleccionadas_modelo.txt").write_text(
    "\n".join(best_columns),
    encoding="utf-8"
)
model.fit(training[best_columns], training["RENDIMIENTO_T_HA"])
prediction_output = prediction[["ID_POLIGONO", "AREA_HA"]].copy()
prediction_output["modelo_elegido"] = best_name
prediction_output["rendimiento_predicho"] = model.predict(prediction[best_columns])
prediction_output.to_csv(
    "predicciones_modelo_completo.csv",
    index=False,
    encoding="utf-8-sig"
)
print("\nModelo elegido por MAE CV:", best_name)
print("Parcelas predichas:", len(prediction_output))
print("Archivo guardado: predicciones_modelo_completo.csv")
