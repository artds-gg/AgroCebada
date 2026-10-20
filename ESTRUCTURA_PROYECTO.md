# Estructura del proyecto AgroCebada

## Propósito

El proyecto desarrolla un modelo de regresión para estimar el rendimiento de parcelas de cebada a partir de observaciones satelitales, variables climáticas, información topográfica y datos de rendimiento disponibles para el conjunto de entrenamiento.

La solución tiene dos componentes:

1. Un flujo reproducible de preparación de datos, selección de variables, entrenamiento y generación de resultados.
2. Una aplicación Streamlit que carga el modelo entrenado, consulta predicciones existentes y permite calcular predicciones para nuevos valores.

## Estructura de carpetas

```text
FIRA/
├── app.py
├── requirements.txt
├── 01_analisis_PRO.ipynb
├── compare_all_features.py
├── compare_basico_sentinel.py
├── final_model_pipeline.py
├── prepare_qgis_dashboard.py
├── DOCUMENTACION_PROYECTO_AGROCEBADA.md
├── ESTRUCTURA_PROYECTO.md
├── REPORTE_TECNICO_AGROCEBADA.tex
├── modelo_compacto.joblib
├── modelo_compacto_metadata.json
├── features_modelo_2025.csv
├── features_seleccionadas_modelo.txt
├── evaluacion_modelo_compacto.csv
├── predicciones_modelo_robusto.csv
├── DATASET_RETO_AGRO_2026/
│   ├── Conjunto_datos_PRO_AgroCebada.csv
│   ├── Conjunto_datos_BASICO_AgroCebada2026.csv
│   ├── ID_area_rendimiento_70_30_Reto_AgroCebada.csv
│   ├── Parcelas_Reto_AGC_CONJUNTO_70_30/
│   ├── Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/
│   ├── Reto_AgroCebada_Temperatura_2022_2025_RECORTADA_CHIRPS/
│   └── Reto_AgroCebada_Topografia_INEGI_CEM4/
└── salidas_visualizacion/
    ├── dashboard_resultados_actualizado.html
    ├── resultados_agrocebada_actualizado.gpkg
    ├── resultados_parcelas_actualizado.csv
    └── resumen_municipio_actualizado.csv
```

## Archivos principales

### Aplicación y despliegue

- `app.py`: aplicación web Streamlit. Carga `modelo_compacto.joblib`, muestra las predicciones existentes y ejecuta predicciones nuevas con las cinco variables aceptadas por el modelo.
- `requirements.txt`: dependencias Python para ejecutar la aplicación.
- `modelo_compacto.joblib`: pipeline entrenado que debe permanecer junto a `app.py`.
- `modelo_compacto_metadata.json`: metadatos del modelo, variables utilizadas y finalidad de la predicción.

### Datos y features

- `DATASET_RETO_AGRO_2026/`: datos originales entregados para el reto y fuentes raster utilizadas en la construcción de variables espaciales.
- `features_modelo_2025.csv`: tabla agregada por parcela con las variables utilizadas durante el modelado.
- `features_seleccionadas_modelo.txt`: lista ordenada de variables seleccionadas para el flujo de entrenamiento.
- `ID_area_rendimiento_70_30_Reto_AgroCebada.csv`: rendimiento conocido para el 70% y partición de parcelas del reto.

### Scripts reproducibles

- `01_analisis_PRO.ipynb`: exploración inicial, limpieza y construcción de la línea base.
- `compare_basico_sentinel.py`: comparación de variables PRO, Sentinel-2 y Landsat.
- `compare_all_features.py`: extracción de variables satelitales, climáticas y topográficas a nivel de parcela.
- `final_model_pipeline.py`: validación aleatoria y espacial, entrenamiento del pipeline robusto y generación de `predicciones_modelo_robusto.csv`.
- `prepare_qgis_dashboard.py`: generación de las salidas geográficas y tablas de visualización. El GeoPackage se conserva como producto geoespacial; no forma parte de la aplicación Streamlit.

### Documentación y resultados

- `DOCUMENTACION_PROYECTO_AGROCEBADA.md`: memoria técnica del análisis, decisiones metodológicas y resultados.
- `ESTRUCTURA_PROYECTO.md`: esta guía de organización y ejecución del repositorio.
- `REPORTE_TECNICO_AGROCEBADA.tex`: reporte técnico para la entrega del reto en formato LaTeX.
- `evaluacion_modelo_compacto.csv`: métricas del modelo compacto, incluyendo validación espacial.
- `predicciones_modelo_robusto.csv`: predicciones, incertidumbre bootstrap, intervalos P10-P90 y confianza relativa.
- `salidas_visualizacion/`: resultados finales para consulta tabular y geoespacial.

## Flujo de datos

```text
Datos originales
      │
      ├── PRO + tabla de rendimiento
      ├── BASICO: Sentinel-2 y Landsat
      ├── CHIRPS: precipitación y temperatura
      └── INEGI: elevación y pendiente
      │
      ▼
Agregación por ID_POLIGONO
      │
      ▼
features_modelo_2025.csv
      │
      ├── Validación y selección de variables
      ├── modelo_compacto.joblib ──► app.py
      └── final_model_pipeline.py ──► predicciones_modelo_robusto.csv
                                                   │
                                                   ▼
                                      salidas_visualizacion/
```

## Modelo utilizado por la aplicación

La aplicación recibe cinco variables numéricas:

| Nombre | Interpretación | Unidad |
|---|---|---|
| `prec_prec_2025_05` | precipitación de mayo de 2025 | mm |
| `prec_prec_2025_02` | precipitación de febrero de 2025 | mm |
| `prec_prec_2025_04` | precipitación de abril de 2025 | mm |
| `tmin_std` | desviación estándar de temperatura mínima | °C |
| `nubosidad_max` | nubosidad máxima observada | % |

El pipeline guardado es un `GradientBoostingRegressor` encapsulado en un pipeline de scikit-learn. La aplicación no reentrena el modelo: carga el artefacto y ejecuta `predict` con los valores enviados por el usuario.

Los valores iniciales del formulario son las medianas del subconjunto de entrenamiento de `features_modelo_2025.csv`. La aplicación limita cada entrada al rango observado durante el entrenamiento y rechaza valores como cero cuando están fuera de ese dominio. Esto evita producir predicciones con datos que no representan una parcela válida.

El archivo `Conjunto_datos_BASICO_AgroCebada2026.csv` no puede copiarse directamente al formulario. Sus columnas son observaciones satelitales crudas; primero deben agregarse por parcela y combinarse con las variables climáticas para producir las cinco entradas que espera el modelo compacto.

## Ejecución local

Desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`.

## Publicación en GitHub y hosting

El repositorio debe contener el código, los archivos de dependencias, el modelo entrenado, sus metadatos, las features requeridas por la aplicación y las predicciones finales. Los archivos temporales de comparación, depuración y validación histórica no son necesarios para ejecutar el servicio y deben permanecer fuera del repositorio activo.

Comando de inicio para un servicio compatible con Streamlit:

```text
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

No deben incluirse claves, tokens, credenciales ni información personal. Si el repositorio se publica como público, los datasets deben revisarse para confirmar que su redistribución está permitida.

## Reproducibilidad

1. Instalar las dependencias declaradas en `requirements.txt`.
2. Mantener la estructura relativa de `app.py`, `modelo_compacto.joblib`, `features_modelo_2025.csv` y `predicciones_modelo_robusto.csv`.
3. Ejecutar `app.py` para consultar o generar predicciones individuales.
4. Ejecutar `compare_all_features.py` solo cuando se requiera reconstruir las features desde las fuentes originales.
5. Ejecutar `final_model_pipeline.py` cuando se requiera repetir la validación y reconstruir el archivo de predicciones robustas.

## Alcance y limitaciones

- El modelo utiliza variables climáticas de 2025; su interpretación principal corresponde a una estimación al cierre o después del ciclo agrícola.
- La validación espacial es más conservadora que la validación aleatoria y debe priorizarse al evaluar parcelas nuevas.
- Una predicción no equivale a una medición de campo.
- La aplicación valida rangos numéricos, pero no puede detectar por sí sola errores en la extracción geográfica de los datos.
- El modelo compacto acepta únicamente las cinco variables definidas en `modelo_compacto_metadata.json`.
