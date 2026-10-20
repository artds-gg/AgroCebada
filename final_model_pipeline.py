from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

ROOT = Path(r"c:\Users\areth\Documents\FIRA")
DATASET = ROOT / "features_modelo_2025.csv"
SELECTED = ROOT / "features_seleccionadas_modelo.txt"
PARCELS = ROOT / "DATASET_RETO_AGRO_2026" / "Parcelas_Reto_AGC_CONJUNTO_70_30" / "Parcelas_Reto_AGC_CONJUNTO.shp"

features = pd.read_csv(DATASET)
selected = [
    line.strip() for line in SELECTED.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

climate_prefixes = ("prec_", "tmax_", "tmin_", "temperatura_", "amplitud_")
base_columns = [column for column in selected if not column.startswith(climate_prefixes)]


def month_column(prefix, month):
    return f"{prefix}_{month:02d}"


def add_crop_cycle_features(frame):
    output = frame.copy()
    periods = {
        "establecimiento_mayo_junio": [5, 6],
        "vegetativo_julio_agosto": [7, 8],
        "llenado_septiembre_octubre": [9, 10],
        "ciclo_mayo_octubre": [5, 6, 7, 8, 9, 10],
    }
    for name, months in periods.items():
        precipitation = [month_column("prec_prec_2025", month) for month in months]
        tmax = [month_column("tmax_tmax_2025", month) for month in months]
        tmin = [month_column("tmin_tmin_2025", month) for month in months]
        tmean = output[tmax].add(output[tmin]).div(2)
        output[f"precipitacion_{name}"] = output[precipitation].sum(axis=1)
        output[f"temperatura_media_{name}"] = tmean.mean(axis=1)
        output[f"amplitud_termica_{name}"] = output[tmax].sub(output[tmin]).mean(axis=1)
        output[f"grados_dia_base5_{name}"] = (tmean - 5).clip(lower=0).sum(axis=1)
    output["precipitacion_max_mensual_ciclo"] = output[
        [month_column("prec_prec_2025", month) for month in range(5, 11)]
    ].max(axis=1)
    return output


features = add_crop_cycle_features(features)
cycle_columns = [
    column for column in features.columns
    if column.startswith(("precipitacion_", "temperatura_media_", "amplitud_termica_", "grados_dia_"))
    and features[column].notna().any()
]

training = features[features["CONJUNTO"] == "ENTRENAMIENTO"].dropna(
    subset=["RENDIMIENTO_T_HA"]
).copy()
prediction = features[features["CONJUNTO"] == "PREDICCION"].copy()
y = training["RENDIMIENTO_T_HA"]

feature_sets = {
    "actual_completo": list(dict.fromkeys(selected)),
    "ciclo_agricola": list(dict.fromkeys(base_columns + cycle_columns)),
    "actual_mas_ciclo": list(dict.fromkeys(selected + cycle_columns)),
}

estimators = {
    "random_forest": RandomForestRegressor(
        n_estimators=400, min_samples_leaf=4, max_features=0.7, random_state=42, n_jobs=-1
    ),
    "extra_trees": ExtraTreesRegressor(
        n_estimators=400, min_samples_leaf=3, max_features=1.0, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.03, max_depth=2, min_samples_leaf=3, random_state=42
    ),
}


def make_model(estimator):
    return Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("modelo", clone(estimator)),
    ])


random_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
spatial = gpd.read_file(PARCELS).rename(columns={"ID_POLIGON": "ID_POLIGONO"})
spatial_projected = spatial.to_crs("EPSG:6372")
spatial["x"] = spatial_projected.geometry.centroid.x
spatial["y"] = spatial_projected.geometry.centroid.y
spatial = spatial[["ID_POLIGONO", "x", "y"]]
spatial = spatial.merge(training[["ID_POLIGONO"]], on="ID_POLIGONO", how="right")
spatial["grupo_espacial"] = KMeans(n_clusters=5, random_state=42, n_init=20).fit_predict(spatial[["x", "y"]])
spatial_groups = spatial.set_index("ID_POLIGONO").loc[training["ID_POLIGONO"], "grupo_espacial"].to_numpy()
group_cv = GroupKFold(n_splits=5)

results = []
oof_store = {}
for feature_name, columns in feature_sets.items():
    X = training[columns]
    for model_name, estimator in estimators.items():
        model = make_model(estimator)
        random_mae = -cross_val_score(model, X, y, cv=random_cv, scoring="neg_mean_absolute_error", n_jobs=1)
        random_rmse = np.sqrt(-cross_val_score(model, X, y, cv=random_cv, scoring="neg_mean_squared_error", n_jobs=1))
        random_prediction = cross_val_predict(
            model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42),
            method="predict", n_jobs=1
        )
        spatial_prediction = cross_val_predict(model, X, y, cv=group_cv, groups=spatial_groups, method="predict", n_jobs=1)
        results.append({
            "features": feature_name,
            "modelo": model_name,
            "variables": len(columns),
            "MAE_aleatorio": random_mae.mean(),
            "RMSE_aleatorio": random_rmse.mean(),
            "R2_aleatorio": r2_score(y, random_prediction),
            "MAE_espacial": mean_absolute_error(y, spatial_prediction),
            "RMSE_espacial": np.sqrt(mean_squared_error(y, spatial_prediction)),
            "R2_espacial": r2_score(y, spatial_prediction),
        })
        oof_store[(feature_name, model_name)] = {"random": random_prediction, "spatial": spatial_prediction}

results_frame = pd.DataFrame(results).sort_values(["MAE_espacial", "MAE_aleatorio"])
results_frame.to_csv(ROOT / "evaluacion_robusta_modelos.csv", index=False, encoding="utf-8-sig")
print(results_frame.round(3).to_string(index=False))

best = results_frame.iloc[0]
best_key = (best["features"], best["modelo"])
best_columns = feature_sets[best["features"]]
best_model = make_model(estimators[best["modelo"]])
best_model.fit(training[best_columns], y)

for second_name in ["extra_trees", "gradient_boosting"]:
    first = oof_store[(best["features"], "random_forest")]["spatial"]
    second = oof_store[(best["features"], second_name)]["spatial"]
    blend = 0.75 * first + 0.25 * second
    print(f"Mezcla random_forest+{second_name} MAE espacial: {mean_absolute_error(y, blend):.3f}")

prediction_output = prediction[["ID_POLIGONO", "AREA_HA"]].copy()
prediction_output["modelo"] = f"{best['features']}_{best['modelo']}"
point_prediction = best_model.predict(prediction[best_columns])
prediction_output["rendimiento_predicho_t_ha"] = point_prediction.round(4)

rng = np.random.default_rng(42)
bootstrap_predictions = []
for _ in range(100):
    sampled_indices = rng.integers(0, len(training), len(training))
    bootstrap_model = make_model(estimators[best["modelo"]])
    bootstrap_model.fit(training.iloc[sampled_indices][best_columns], y.iloc[sampled_indices])
    bootstrap_predictions.append(bootstrap_model.predict(prediction[best_columns]))
bootstrap_predictions = np.asarray(bootstrap_predictions)
prediction_output["incertidumbre_std_t_ha"] = bootstrap_predictions.std(axis=0).round(4)
prediction_output["intervalo_p10_t_ha"] = np.quantile(bootstrap_predictions, 0.10, axis=0).round(4)
prediction_output["intervalo_p90_t_ha"] = np.quantile(bootstrap_predictions, 0.90, axis=0).round(4)
prediction_output["confianza_relativa"] = pd.cut(
    prediction_output["incertidumbre_std_t_ha"],
    bins=[-np.inf, 0.15, 0.30, np.inf],
    labels=["alta", "media", "baja"]
)
prediction_output.to_csv(ROOT / "predicciones_modelo_robusto.csv", index=False, encoding="utf-8-sig")

spatial_oof = oof_store[best_key]["spatial"]
residuals = y.to_numpy() - spatial_oof
metrics = pd.DataFrame([{
    "features": best["features"], "modelo": best["modelo"],
    "MAE_espacial": mean_absolute_error(y, spatial_oof),
    "RMSE_espacial": np.sqrt(mean_squared_error(y, spatial_oof)),
    "R2_espacial": r2_score(y, spatial_oof),
    "residuo_p10": np.quantile(residuals, 0.10),
    "residuo_p90": np.quantile(residuals, 0.90),
}])
metrics.to_csv(ROOT / "metricas_modelo_robusto.csv", index=False, encoding="utf-8-sig")
print("\nModelo final seleccionado:", best_key)
print("Predicciones generadas:", len(prediction_output))
print("Archivo: predicciones_modelo_robusto.csv")
print(metrics.round(3).to_string(index=False))
