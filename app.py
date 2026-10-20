from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "modelo_compacto.joblib"
METRICS_PATH = ROOT / "evaluacion_modelo_compacto.csv"
PREDICTIONS_PATH = ROOT / "predicciones_modelo_robusto.csv"

FEATURES = [
    "prec_prec_2025_05",
    "prec_prec_2025_02",
    "prec_prec_2025_04",
    "tmin_std",
    "nubosidad_max",
]
LABELS = {
    "prec_prec_2025_05": ("Precipitación de mayo", "mm"),
    "prec_prec_2025_02": ("Precipitación de febrero", "mm"),
    "prec_prec_2025_04": ("Precipitación de abril", "mm"),
    "tmin_std": ("Variabilidad de temperatura mínima", "°C"),
    "nubosidad_max": ("Nubosidad máxima", "%"),
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_data():
    values = pd.read_csv(ROOT / "features_modelo_2025.csv")
    if "CONJUNTO" in values.columns:
        training = values[values["CONJUNTO"] == "ENTRENAMIENTO"]
        if not training.empty:
            values = training
    return values[FEATURES].apply(pd.to_numeric, errors="coerce").dropna()


@st.cache_data
def load_training_stats():
    training = load_training_data()
    return {
        "median": training.median().to_dict(),
        "minimum": training.min().to_dict(),
        "maximum": training.max().to_dict(),
    }


@st.cache_data
def load_predictions():
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH)
    return pd.DataFrame()


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        return metrics.iloc[0].to_dict()
    return {}


def safe_number(value, minimum, maximum):
    if value < minimum or value > maximum:
        raise ValueError(f"El valor debe estar entre {minimum:g} y {maximum:g}.")
    return value


st.set_page_config(
    page_title="AgroCebada | Predicción de rendimiento",
    page_icon="🌾",
    layout="wide",
)

st.title("Predicción de rendimiento de cebada")
st.caption(
    "Consulta las predicciones existentes o introduce valores nuevos para ejecutar el modelo en tiempo real."
)

try:
    model = load_model()
    training_stats = load_training_stats()
except Exception as error:
    st.error(f"No se pudo cargar el modelo o sus datos: {error}")
    st.stop()

metrics = load_metrics()
tab_prediction, tab_existing, tab_about = st.tabs(
    ["Nueva predicción", "Predicciones existentes", "Modelo y límites"]
)

with tab_prediction:
    st.subheader("Introduce las condiciones de la parcela")
    st.info(
        "Los valores iniciales son medianas del conjunto de entrenamiento. La aplicación solo acepta valores dentro del dominio observado por el modelo."
    )
    st.warning(
        "El archivo BASICO contiene observaciones satelitales crudas. No se deben copiar directamente sus columnas a este formulario: las cinco entradas ya son variables agregadas y climáticas."
    )

    with st.form("prediction_form"):
        columns = st.columns(2)
        inputs = {}
        ranges = {
            feature: (
                float(training_stats["minimum"][feature]),
                float(training_stats["maximum"][feature]),
            )
            for feature in FEATURES
        }
        for index, feature in enumerate(FEATURES):
            label, unit = LABELS[feature]
            minimum, maximum = ranges[feature]
            observed_median = float(training_stats["median"][feature])
            step = max((maximum - minimum) / 100, 0.001)
            inputs[feature] = columns[index % 2].number_input(
                f"{label} ({unit})",
                min_value=minimum,
                max_value=maximum,
                value=observed_median,
                step=step,
                format="%.6f",
                help=(
                    f"Variable usada por el modelo: {feature}. "
                    f"Rango observado en entrenamiento: {minimum:.6f} a {maximum:.6f}."
                ),
            )

        submitted = st.form_submit_button("Calcular predicción", type="primary")

    if submitted:
        try:
            for feature, value in inputs.items():
                safe_number(value, *ranges[feature])
            row = pd.DataFrame([inputs], columns=FEATURES)
            prediction = float(model.predict(row)[0])
            st.success("Predicción calculada correctamente")
            result_col, note_col = st.columns([1, 2])
            result_col.metric("Rendimiento estimado", f"{prediction:.3f} t/ha")
            note_col.warning(
                "Esta predicción es una estimación del modelo. Debe validarse con observaciones de campo y no sustituye una medición agronómica."
            )
        except Exception as error:
            st.error(f"No se pudo calcular la predicción: {error}")

with tab_existing:
    st.subheader("Predicciones generadas para las parcelas del proyecto")
    existing = load_predictions()
    if existing.empty:
        st.warning("No se encontró el archivo de predicciones existente.")
    else:
        query = st.text_input("Filtrar por ID de parcela", key="existing_filter")
        if query:
            existing = existing[
                existing["ID_POLIGONO"].astype(str).str.contains(query, case=False, na=False)
            ]
        st.dataframe(existing, use_container_width=True, hide_index=True)

with tab_about:
    st.subheader("Información del modelo")
    st.write(
        "El modelo compacto es un Gradient Boosting entrenado con cinco variables climáticas y de observación."
    )
    if metrics:
        st.metric("MAE de validación espacial", f"{metrics.get('MAE_espacial', 0):.3f} t/ha")
        st.metric("R² de validación espacial", f"{metrics.get('R2_espacial', 0):.3f}")
    st.markdown(
        "**Importante:** la aplicación solo acepta las cinco variables que el modelo conoce. "
        "No se deben introducir ceros para representar datos faltantes; primero deben obtenerse "
        "los valores de la parcela mediante la misma metodología de extracción usada durante el entrenamiento."
    )
