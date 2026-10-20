# Documentación del proyecto de análisis y modelado de rendimiento de cebada

## 1. Objetivo del proyecto

El objetivo general del proyecto es desarrollar una solución funcional para el sector agrícola que permita predecir, visualizar y monitorear el rendimiento de parcelas de cebada a partir de datos satelitales y de rendimiento agronómico.

La línea base inicial se centró en el archivo `Conjunto_datos_PRO_AgroCebada.csv`, porque fue el conjunto más manejable y con mayor claridad para construir un primer modelo supervisado. Posteriormente, se mejoró la línea base incorporando señales temporales del comportamiento de vegetación por parcela.

## 2. Archivos del proyecto
- `DATASET_RETO_AGRO_2026/Conjunto_datos_PRO_AgroCebada.csv`: conjunto principal de trabajo con observaciones satelitales, especialmente NDVI, EVI, LAI y MSAVI.
- `DATASET_RETO_AGRO_2026/Conjunto_datos_BASICO_AgroCebada2026.csv`: conjunto más completo y con más historial, pero más complejo; se recomienda incorporarlo más adelante, por sensor y de forma controlada.
- `DATASET_RETO_AGRO_2026/ID_area_rendimiento_70_30_Reto_AgroCebada.csv`: tabla objetivo con rendimiento y partición de entrenamiento/predicción.
- `01_analisis_PRO.ipynb`: cuaderno principal de análisis exploratorio, limpieza y modelado.

## 3. Ruta de trabajo seguida
### Fase 1: Reconocimiento de la estructura del proyecto

Se revisó la carpeta del proyecto para identificar qué archivos existían, qué formato tenían y cuál era el conjunto más útil para comenzar. Aquí la idea fue evitar mezclar todos los datos desde el inicio y trabajar primero sobre la fuente más directa del problema.

### Fase 2: Diagnóstico de calidad de datos

Se revisó la calidad del archivo PRO y se detectó que no era un problema de “series temporales perfectas”, sino un panel temporal irregular con observaciones por parcela y por fecha. Las decisiones clave fueron:

- Ordenar por `ID_POLIGONO` y `fecha_captura`.
- Convertir columnas numéricas a tipo numérico con coerción.
- Identificar filas con nubosidad total (100%) como no útiles.
- Detectar valores inválidos como `LAI` negativo y convertirlos en nulos.
- Separar el análisis en datos útiles y no útiles sin destruir la estructura original.

Esto fue importante porque el problema real no era un dataset completamente limpio, sino un dataset con observaciones parciales y ruido temporal. La capacidad de distinguir “dato útil” de “dato no observacional” fue clave para mantener la integridad del análisis.

### Fase 3: EDA y comprensión de la variable objetivo

Se analizó la distribución del rendimiento, la variabilidad entre parcelas y la relación de las métricas satelitales con la producción. Se comprobó que el rendimiento era la variable central del problema, y que un enfoque por parcela era más racional que manejar cada observación temporal como punto independiente.

### Fase 4: Agregado por parcela

El problema de regresión se reformuló a nivel de parcela:

- Se agruparon observaciones por `ID_POLIGONO`.
- Se calcularon métricas como media, máximo, mínimo, desviación y tendencia del NDVI, EVI, LAI y MSAVI.
- Se mantuvo la información del área y del rendimiento objetivo.

Este paso fue crucial porque la producción final se asocia a la parcela, no a la observación individual. La mayor parte del ruido se elimina al resumir la evolución temporal de la vegetación en indicadores agregados de cada parcela.

### Fase 5: Modelo base

Se entrenaron varios modelos con un conjunto de variables agregadas simples:

- media NDVI
- máximo NDVI
- mínimo NDVI
- desviación NDVI
- media EVI
- media LAI
- media MSAVI
- nubosidad media
- observaciones válidas
- área

Los modelos comparados fueron:

- media de referencia
- regresión lineal
- Random Forest

La línea base mostró que el modelo podía aprender algo, pero aún era limitado porque solo capturaba el nivel general de la vegetación y no su evolución a lo largo del ciclo.

## 4. Por qué se mejoró el modelo

La mejora se fundamentó en un punto clave: el rendimiento agrícola depende no solo del nivel promedio de vegetación, sino también de su evolución temporal. Por ejemplo:

- un NDVI alto al final del ciclo puede tener un significado muy distinto a un NDVI alto desde el inicio;
- una parcela con fuerte crecimiento o ganancia sostenida puede rendir mejor que otra con un nivel aparentemente similar pero sin tendencia;
- un aumento o decremento consistente en LAI y MSAVI puede reflejar mejor la madurez y el vigor de la vegetación.

Por esta razón, se agregaron variables como:

- `ndvi_ultimo`
- `ndvi_delta`
- `ndvi_trend`
- `evi_ultimo`
- `lai_ultimo`
- `msavi_ultimo`
- `dias_observados`

Esto mejora la capacidad del modelo para capturar la trayectoria del crecimiento y no solo el estado puntual.

## 5. Mejoras aplicadas y validación

Se validó el enfoque mejorado con resultados reproducibles. La ejecución mostró esto:

- Extra Trees: MAE = 0.421, RMSE = 0.646, R² = 0.133
- Random Forest: MAE = 0.464, RMSE = 0.641, R² = 0.147
- HistGradientBoosting: MAE = 0.487, RMSE = 0.731, R² = -0.112
- Regresión lineal: MAE = 0.672, RMSE = 0.888, R² = -0.640
- Media de referencia: MAE = 0.675, RMSE = 0.774, R² = -0.245

En validación cruzada, el mejor desempeño volvió a corresponder a Extra Trees y Random Forest:

- Extra Trees: MAE_CV_media = 0.434, RMSE_CV_media = 0.602
- Random Forest: MAE_CV_media = 0.450, RMSE_CV_media = 0.614

Esto confirma que la mejora no es solo un artefacto del conjunto de prueba; se observa también en validación cruzada y con una estructura más informativa.

## 6. Resultado actual del proyecto

Hasta este momento, se ha logrado lo siguiente:

- Comprender el formato y la utilidad de cada archivo.
- Diagnosticar la calidad de los datos de manera realista.
- Limpiar y transformar el problema a una escala de parcela.
- Construir una línea base funcional.
- Mejorar la línea base con variables temporales y validación cruzada.
- Generar una base sólida para una siguiente integración del conjunto BÁSICO, con un criterio controlado por sensor y por tipo de variable.

## 7. Recomendación para continuar

La estrategia más sensata es la siguiente:

1. Mantener la mejora actual de PRO como línea base sólida.
2. Incorporar BÁSICO por separado, no todos los sensores mezclados.
3. Primero evaluar Landsat y Sentinel-2 por separado.
4. Agregar variables climáticas y topográficas solo después de que el modelo PRO esté bien reforzado.
5. Comparar si cada nueva capa realmente mejora la capacidad predictiva.

Esto reduce el riesgo de introducir ruido, confusión y overfitting antes de tener una base estable.

## 8. Conclusión

El proyecto pasó de una etapa de diagnóstico y limpieza a una etapa de modelado con mayor rigor. La gran mejora no viene de “hacer más modelos”, sino de transformar correctamente la representación del fenómeno: pasar de observaciones temporales aisladas a indicadores de crecimiento y evolución por parcela. Esa decisión es la que explica la mejora real del rendimiento predictivo.

## 9. Diagnóstico del conjunto BÁSICO

Antes de integrarlo al modelo se midieron sus claves, sensores, fechas y cobertura de índices.

Resultados:

- 107,666 filas.
- 197 parcelas.
- Coincidencia de 197 de 197 parcelas con la tabla objetivo.
- Sensores: 48,626 filas de Landsat y 59,040 de Sentinel-2.
- Periodo: 2022-2025.
- Filas con algún índice disponible: 31,048 para Landsat y 44,877 para Sentinel-2.

La cobertura no es homogénea:

- Sentinel-2 tiene aproximadamente 76% de cobertura para los índices principales del archivo.
- Landsat no tiene cobertura útil para la mayoría de esos índices; el índice con cobertura apreciable es `VI6T`.

Por esta razón, no se deben combinar ambos sensores como si fueran equivalentes. La siguiente prueba debe comparar PRO contra PRO + Sentinel-2, mientras que Landsat se mantendrá como una fuente independiente para una evaluación específica de `VI6T`.

La decisión también evita convertir valores faltantes estructurales en ceros o en supuestas observaciones de baja vegetación.

## 10. Comparación de sensores

Se amplió `compare_basico_sentinel.py` para comparar, usando las mismas 138 parcelas de entrenamiento, las features temporales completas de PRO y validación cruzada de cinco pliegues:

- PRO 2025 solamente: MAE promedio = 0.508 y RMSE promedio = 0.653.
- PRO 2025 + Sentinel-2 2025: MAE promedio = 0.490 y RMSE promedio = 0.638.
- PRO 2025 + Landsat VI6T 2025: MAE promedio = 0.454 y RMSE promedio = 0.598.

En esta comparación, Landsat + `VI6T` redujo el MAE aproximadamente 10.6% respecto a PRO 2025 y superó al candidato con Sentinel-2. El resultado es prometedor, pero debe revisarse con validación repetida y análisis de cobertura antes de adoptarlo definitivamente.

Landsat no se incorpora en esta etapa porque sus índices no tienen cobertura comparable en el archivo BÁSICO. Se evaluará de forma independiente usando las variables que realmente estén disponibles, especialmente `VI6T`.

La comparación histórica de sensores se utilizó para seleccionar las variables del modelo final. Sus archivos temporales ya no se conservan en la carpeta activa.

## 11. Ingeniería de variables climáticas y topográficas

Se creó `compare_all_features.py` para extraer variables raster a nivel de parcela. La extracción usa el promedio zonal de los píxeles que toca cada polígono (`all_touched=True`). Esta decisión es necesaria porque los raster climáticos tienen resolución aproximada de 0.05 grados y algunas parcelas son más pequeñas que un píxel; usar únicamente el centro del píxel habría producido valores faltantes artificiales.

Se incorporaron precipitación mensual, temperatura máxima y mínima de 2025, además de sus resúmenes estadísticos, temperatura media anual, amplitud térmica, elevación y pendiente.

La validación repetida utilizó 5 pliegues repetidos 5 veces sobre las 138 parcelas de entrenamiento:

- PRO: MAE = 0.510, RMSE = 0.660.
- PRO + Landsat: MAE = 0.454, RMSE = 0.601.
- PRO + Landsat + Sentinel-2: MAE = 0.464, RMSE = 0.612.
- PRO + Landsat + Sentinel-2 + clima: MAE = 0.389, RMSE = 0.528.
- PRO + Landsat + Sentinel-2 + clima + topografía: MAE = 0.392, RMSE = 0.530.

El mejor candidato actual es `pro_landsat_sentinel_clima`. La topografía se conserva como bloque experimental porque no mejoró el error en esta comparación.

El script genera archivos intermedios durante su ejecución; la carpeta activa conserva únicamente las features finales necesarias para el modelo.

## 12. Optimización, estabilidad e incertidumbre

La optimización histórica comparó configuraciones de Extra Trees, Random Forest y Gradient Boosting con validación cruzada repetida. El resultado final conservado es el modelo compacto utilizado por la aplicación.

Resultados de la optimización:

- MAE promedio repetido: 0.378.
- RMSE promedio repetido: 0.518.
- MAE fuera de muestra con una partición de validación cruzada: 0.376.
- RMSE fuera de muestra: 0.535.
- R² fuera de muestra: 0.602.

También se realizó una ablación para verificar la estabilidad de la precipitación de mayo:

- Modelo completo: MAE = 0.376.
- Sin precipitación de mayo: MAE = 0.398.
- Sin precipitación mensual: MAE = 0.401.
- Sin variables climáticas: MAE = 0.492.

La precipitación de mayo es importante, pero su efecto se comprobó con 50 particiones repetidas y no solamente con una importancia calculada sobre el ajuste final.

Las predicciones optimizadas fueron un resultado intermedio de validación. La salida operativa conservada es `predicciones_modelo_robusto.csv`; sus intervalos deben interpretarse como orientativos.

### Alcance temporal

El modelo actual utiliza variables climáticas de todo 2025. Por tanto, es un modelo de estimación del rendimiento al cierre o después del ciclo agrícola. Para una predicción anticipada habrá que entrenar versiones con cortes temporales, por ejemplo hasta marzo, abril o mayo, eliminando toda variable posterior al corte.

## 13. Validación para aplicar a otras parcelas

Se construyó `final_model_pipeline.py` para comprobar si el resultado se mantiene cuando se dejan grupos geográficos completos fuera del entrenamiento. Se usaron grupos espaciales creados a partir de los centroides proyectados de las parcelas.

La validación aleatoria produjo errores cercanos a 0.38 MAE, pero la validación espacial fue más exigente:

- Mejor candidato espacial: `actual_completo + gradient_boosting`.
- MAE espacial: 0.638 t/ha.
- RMSE espacial: 0.814 t/ha.
- R² espacial: 0.078.

Las variables de ciclo agrícola mayo-octubre no mejoraron la extrapolación espacial en esta muestra. Por tanto, no se deben presentar las métricas aleatorias como garantía de funcionamiento en parcelas completamente nuevas; la métrica espacial es la referencia conservadora para generalización.

El archivo `predicciones_modelo_robusto.csv` incluye predicción puntual a cuatro decimales, desviación estándar bootstrap, intervalo P10-P90 y nivel de confianza relativo. Para reducir decisiones incorrectas, las parcelas con confianza baja deben revisarse antes de usarse operativamente.

## 14. Precisión comparada con parcelas conocidas

La precisión se calculó con predicciones fuera de muestra sobre las 138 parcelas con rendimiento real. No se comparó el modelo contra las mismas observaciones usadas para entrenarlo.

| Validación | MAE | RMSE | MAPE | Dentro de ±10% | Dentro de ±20% | Dentro de ±0.5 t/ha |
|---|---:|---:|---:|---:|---:|---:|
| Aleatoria | 0.409 t/ha | 0.577 t/ha | 11.7% | 65.2% | 85.5% | 72.5% |
| Espacial | 0.638 t/ha | 0.814 t/ha | 19.6% | 42.0% | 63.8% | 50.7% |

La validación aleatoria describe el comportamiento cuando las parcelas nuevas son similares y cercanas a las conocidas. La validación espacial es más conservadora y representa mejor el caso de aplicar el modelo a otra zona. Por eso no debe comunicarse un único porcentaje de confianza sin especificar el tipo de validación.

Estas métricas corresponden a la validación histórica del modelo. En un problema de regresión, “falso positivo” y “falso negativo” no son categorías directas; se controlan mediante error absoluto, sesgo, intervalos y tolerancias de rendimiento.

## 15. Visualización y uso de resultados

Se generaron salidas geográficas y una página HTML local mediante `prepare_qgis_dashboard.py`. La estructura de ejecución y publicación está documentada en [ESTRUCTURA_PROYECTO.md](ESTRUCTURA_PROYECTO.md). El GeoPackage actualizado es [resultados_agrocebada_actualizado.gpkg](salidas_visualizacion/resultados_agrocebada_actualizado.gpkg), con capas de parcelas y resumen municipal.

