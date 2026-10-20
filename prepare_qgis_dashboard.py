from pathlib import Path
import json
import html

import geopandas as gpd
import pandas as pd

ROOT = Path(r"c:\Users\areth\Documents\FIRA")
OUTPUT = ROOT / "salidas_visualizacion"
OUTPUT.mkdir(exist_ok=True)

PARCELS = ROOT / "DATASET_RETO_AGRO_2026" / "Parcelas_Reto_AGC_CONJUNTO_70_30" / "Parcelas_Reto_AGC_CONJUNTO.shp"
TARGET = ROOT / "DATASET_RETO_AGRO_2026" / "ID_area_rendimiento_70_30_Reto_AgroCebada.csv"
PREDICTIONS = ROOT / "predicciones_modelo_robusto.csv"

parcels = gpd.read_file(PARCELS).rename(columns={"ID_POLIGON": "ID_POLIGONO"})
parcels = parcels.drop(columns=["CONJUNTO"], errors="ignore")
target = pd.read_csv(TARGET)
predictions = pd.read_csv(PREDICTIONS)

keep_target = ["ID_POLIGONO", "AREA_HA", "RENDIMIENTO_T_HA", "CONJUNTO"]
layer = parcels.merge(target[keep_target], on="ID_POLIGONO", how="left")
layer = layer.merge(predictions, on=["ID_POLIGONO", "AREA_HA"], how="left")
layer["tipo_parcela"] = layer["CONJUNTO"].map({
    "ENTRENAMIENTO": "rendimiento_conocido",
    "PREDICCION": "rendimiento_por_predecir",
}).fillna("sin_clasificar")
layer["estado_resultado"] = layer["rendimiento_predicho_t_ha"].notna().map({
    True: "prediccion_disponible",
    False: "sin_prediccion_final",
})
layer["error_referencia_t_ha"] = (
    layer["RENDIMIENTO_T_HA"] - layer["rendimiento_predicho_t_ha"]
).abs()
layer["alerta_revision"] = layer["confianza_relativa"].isin(["baja", "media"])

summary = (
    layer.groupby(["Estado", "Municipio"], dropna=False)
    .agg(
        parcelas=("ID_POLIGONO", "count"),
        parcelas_prediccion=("rendimiento_predicho_t_ha", "count"),
        rendimiento_real_medio=("RENDIMIENTO_T_HA", "mean"),
        rendimiento_predicho_medio=("rendimiento_predicho_t_ha", "mean"),
        incertidumbre_media=("incertidumbre_std_t_ha", "mean"),
    )
    .reset_index()
)

# QGIS layers: original CRS is retained and fields are simple scalar attributes.
layer.to_file(OUTPUT / "resultados_agrocebada_actualizado.gpkg", layer="parcelas_resultados", driver="GPKG")
summary_gdf = parcels.dissolve(by=["Estado", "Municipio"], as_index=False).merge(
    summary, on=["Estado", "Municipio"], how="left"
)
summary_gdf.to_file(OUTPUT / "resultados_agrocebada_actualizado.gpkg", layer="resumen_municipio", driver="GPKG")

layer.drop(columns="geometry").to_csv(
    OUTPUT / "resultados_parcelas_actualizado.csv", index=False, encoding="utf-8-sig"
)
summary.to_csv(
    OUTPUT / "resumen_municipio_actualizado.csv", index=False, encoding="utf-8-sig"
)

records = layer.drop(columns="geometry").where(pd.notna(layer.drop(columns="geometry")), None).to_dict(orient="records")
for record in records:
    for key, value in record.items():
        if hasattr(value, "item"):
            record[key] = value.item()

data_json = json.dumps(records, ensure_ascii=False)
summary_rows = "".join(
    f"<tr><td>{html.escape(str(row['Estado']))}</td><td>{html.escape(str(row['Municipio']))}</td>"
    f"<td>{int(row['parcelas'])}</td><td>{int(row['parcelas_prediccion'])}</td>"
    f"<td>{row['rendimiento_predicho_medio']:.3f}</td><td>{row['incertidumbre_media']:.3f}</td></tr>"
    for _, row in summary.iterrows()
)
mean_prediction = layer["rendimiento_predicho_t_ha"].mean()
mean_uncertainty = layer["incertidumbre_std_t_ha"].mean()
predicted_count = layer["rendimiento_predicho_t_ha"].notna().sum()

html_page = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Resultados AgroCebada</title>
<style>
body{{font-family:Arial,sans-serif;background:#f3f6f4;color:#17221d;margin:0}}
main{{max-width:1180px;margin:32px auto;padding:0 20px}}
h1{{margin-bottom:6px}}.muted{{color:#5e6b63}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}
.card{{background:white;border:1px solid #d8e0da;border-radius:8px;padding:18px}}
.value{{font-size:28px;font-weight:700;margin-top:8px}}
.panel{{background:white;border:1px solid #d8e0da;border-radius:8px;padding:18px;margin-top:18px;overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #e6ebe7;text-align:left}}th{{background:#eef4ef;position:sticky;top:0}}
input{{width:100%;max-width:420px;padding:10px;border:1px solid #bbc9be;border-radius:5px;margin-bottom:12px}}
@media(max-width:700px){{.cards{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<h1>Resultados de rendimiento de cebada</h1>
<p class="muted">Modelo robusto: Gradient Boosting con PRO, Landsat, Sentinel-2 y clima. Las predicciones corresponden a parcelas del conjunto de predicción.</p>
<section class="cards"><div class="card">Parcelas predichas<div class="value">{predicted_count}</div></div><div class="card">Rendimiento medio (t/ha)<div class="value">{mean_prediction:.3f}</div></div><div class="card">Incertidumbre media (t/ha)<div class="value">{mean_uncertainty:.3f}</div></div></section>
<section class="panel"><h2>Resumen municipal</h2><table><thead><tr><th>Estado</th><th>Municipio</th><th>Parcelas</th><th>Predicciones</th><th>Rendimiento medio</th><th>Incertidumbre</th></tr></thead><tbody>{summary_rows}</tbody></table></section>
<section class="panel"><h2>Parcelas</h2><input id="search" placeholder="Filtrar por ID, estado o municipio"><table id="parcel-table"><thead><tr><th>ID</th><th>Estado</th><th>Municipio</th><th>Tipo</th><th>Predicción t/ha</th><th>P10</th><th>P90</th><th>Confianza</th></tr></thead><tbody></tbody></table></section>
</main>
<script>
const rows={data_json};
const body=document.querySelector('#parcel-table tbody'); const search=document.querySelector('#search');
function render(){{const q=search.value.toLowerCase(); body.innerHTML=rows.filter(r=>r.estado_resultado==='prediccion_disponible' && JSON.stringify(r).toLowerCase().includes(q)).map(r=>`<tr><td>${{r.ID_POLIGONO??''}}</td><td>${{r.Estado??''}}</td><td>${{r.Municipio??''}}</td><td>${{r.tipo_parcela??''}}</td><td>${{r.rendimiento_predicho_t_ha??''}}</td><td>${{r.intervalo_p10_t_ha??''}}</td><td>${{r.intervalo_p90_t_ha??''}}</td><td>${{r.confianza_relativa??''}}</td></tr>`).join('')}}
search.addEventListener('input',render); render();
</script></body></html>"""
(OUTPUT / "dashboard_resultados_actualizado.html").write_text(html_page, encoding="utf-8")
print("Salidas creadas en:", OUTPUT)
print("Parcelas en capa:", len(layer))
print("Predicciones con incertidumbre:", layer["rendimiento_predicho_t_ha"].notna().sum())
