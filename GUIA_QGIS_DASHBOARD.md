# Guía para visualizar e interpretar resultados

## Archivos creados

La carpeta `salidas_visualizacion` contiene:

- `resultados_agrocebada_actualizado.gpkg`: GeoPackage con las capas `parcelas_resultados` y `resumen_municipio`.
- `resultados_parcelas_actualizado.csv`: tabla completa de parcelas.
- `resumen_municipio_actualizado.csv`: resumen agregado por municipio.
- `dashboard_resultados_actualizado.html`: página local para revisar resultados sin instalar una aplicación.

## Abrir resultados en QGIS

1. Abrir QGIS.
2. Seleccionar `Capa` > `Añadir capa` > `Añadir capa vectorial`.
3. Seleccionar `salidas_visualizacion/resultados_agrocebada_actualizado.gpkg`.
4. En la lista de capas elegir `parcelas_resultados`.
5. Añadir también `resumen_municipio` si se desea una vista agregada.

## Simbología recomendada

Para la capa `parcelas_resultados`:

1. Abrir `Propiedades` de la capa.
2. Ir a `Simbología`.
3. Elegir `Graduado`.
4. Usar el campo `rendimiento_predicho_t_ha`.
5. Usar 5 clases y una rampa de color secuencial.
6. Aplicar.

Para observar incertidumbre:

- usar `incertidumbre_std_t_ha` como campo de clasificación;
- usar `confianza_relativa` para filtrar `alta`, `media` o `baja`;
- usar `alerta_revision = TRUE` para localizar parcelas que requieren revisión.

Para distinguir el conjunto:

- `tipo_parcela = rendimiento_conocido`: parcela con rendimiento real;
- `tipo_parcela = rendimiento_por_predecir`: parcela cuya producción se estima.

## Campos principales

- `rendimiento_predicho_t_ha`: estimación puntual en t/ha.
- `intervalo_p10_t_ha`: límite inferior orientativo.
- `intervalo_p90_t_ha`: límite superior orientativo.
- `incertidumbre_std_t_ha`: dispersión del bootstrap.
- `confianza_relativa`: clasificación alta, media o baja.
- `RENDIMIENTO_T_HA`: rendimiento real, disponible para entrenamiento.
- `error_referencia_t_ha`: error absoluto, cuando hay valor real y predicción comparable.
- `estado_resultado`: `prediccion_disponible` para las 59 parcelas objetivo y `sin_prediccion_final` para las 138 parcelas de entrenamiento.

## Por qué aparecen valores vacíos

Los valores vacíos son intencionales:

- `RENDIMIENTO_T_HA` está vacío en las 59 parcelas de predicción porque ese rendimiento aún no se conoce.
- Las columnas de predicción están vacías en las 138 parcelas de entrenamiento porque el archivo final se generó para las 59 parcelas objetivo.
- `error_referencia_t_ha` no se calcula en esta capa porque no existe una pareja de rendimiento real y predicción final para la misma parcela.

Para evaluar el entrenamiento se deben usar los archivos de validación fuera de muestra, no rellenar estas celdas con cero.

## Dashboard local

Abrir directamente en el navegador:

`salidas_visualizacion/dashboard_resultados_actualizado.html`

La página permite:

- consultar el rendimiento medio por municipio;
- filtrar parcelas por ID, estado o municipio;
- revisar predicción, intervalo y confianza;
- identificar parcelas con incertidumbre mayor.

## Interpretación recomendada

- No ordenar las parcelas solo por predicción puntual.
- Revisar también el intervalo P10-P90.
- Priorizar inspección de parcelas con confianza baja.
- Comparar predicción municipal y distribución espacial.
- No interpretar la predicción como una medición de campo.

## Convertirlo en una aplicación

La página HTML sirve como prototipo local. Para una aplicación formal se puede migrar a:

- Streamlit: dashboard analítico rápido en Python;
- Dash: dashboard con más controles y gráficos;
- aplicación web con FastAPI y React;
- QGIS Server o QGIS Cloud para publicación geoespacial.

La fuente de datos de la aplicación debe ser `resultados_parcelas_actualizado.csv` o el GeoPackage actualizado, y el modelo debe conservar su versión, fecha de entrenamiento y métricas de validación espacial.

## Aplicación interactiva en línea

El archivo `dashboard_resultados_actualizado.html` conserva la vista estática y no puede ejecutar Python en el navegador. Para permitir nuevas predicciones se agregó `app.py`, una aplicación Streamlit que:

- conserva la consulta de predicciones existentes;
- permite introducir valores nuevos para las cinco variables que conoce el modelo;
- ejecuta `modelo_compacto.joblib` en el servidor;
- valida rangos de entrada y muestra la predicción en t/ha;
- muestra las métricas de validación y las limitaciones del modelo.

### Ejecutar localmente

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Publicar

Subir al repositorio como mínimo `app.py`, `requirements.txt`, `modelo_compacto.joblib`, `modelo_compacto_metadata.json`, `features_modelo_2025.csv`, `predicciones_modelo_robusto.csv` y `evaluacion_modelo_compacto.csv`. En el proveedor elegido, configurar el comando de inicio como:

```text
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

No se deben publicar claves, credenciales ni archivos con datos sensibles. Antes de ponerlo en producción conviene proteger el acceso, limitar el tamaño de entradas y registrar la versión del modelo utilizada en cada predicción.
