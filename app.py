import os

import joblib
import pandas as pd
import streamlit as st

# Configurar la página de Streamlit
st.set_page_config(page_title="Predicción de Aprobación de Curso", layout="centered")

st.title("Predicción de Nota Final - Curso")
st.write(
    "Introduce los datos del estudiante para estimar su nota final "
    "utilizando el modelo optimizado de Bagging."
)

# Ruta base: carpeta donde está este archivo (funciona local y en Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 1. Cargar artefactos necesarios de forma segura
@st.cache_resource
def load_artifacts():
    columnas = joblib.load(os.path.join(BASE_DIR, "one_hot_columns.joblib"))
    scaler = joblib.load(os.path.join(BASE_DIR, "min_max_scaler.joblib"))
    model = joblib.load(os.path.join(BASE_DIR, "bagging_optimizado.joblib"))
    return list(columnas), scaler, model


try:
    columnas_modelo, scaler, model = load_artifacts()
except Exception as e:
    st.error(f"Error al cargar los archivos .joblib: {e}")
    st.warning(
        "Asegúrate de que 'one_hot_columns.joblib', 'min_max_scaler.joblib' y "
        "'bagging_optimizado.joblib' estén en la misma carpeta que app.py."
    )
    st.stop()

# 2. Formulario de entrada de usuario
st.header("Datos del Estudiante")

# Categorías de Felder presentes en el modelo. 'activo' fue la categoría
# eliminada en el one-hot (drop_first), por eso se agrega como referencia.
categorias_felder = ["activo"] + [
    col.replace("Felder_", "") for col in columnas_modelo if col.startswith("Felder_")
]

felder_selected = st.selectbox("Estilo de Aprendizaje (Felder)", options=categorias_felder)
examen_admision = st.slider(
    "Nota de Examen de Admisión", min_value=0.0, max_value=5.0, value=3.8, step=0.05
)

if st.button("Calcular Predicción"):
    # 3. Procesar datos de entrada igual que en el entrenamiento
    fila = {}
    for col in columnas_modelo:
        if col.startswith("Felder_"):
            fila[col] = 1.0 if col == f"Felder_{felder_selected}" else 0.0

    # Min-Max Scaler (fue ajustado con la columna 'Examen_admisión')
    examen_df = pd.DataFrame({"Examen_admisión": [examen_admision]})
    fila["Examen_admision_scaled"] = float(scaler.transform(examen_df)[0][0])

    # Ordenar las columnas exactamente como las espera el modelo
    df_procesado = pd.DataFrame([fila])[columnas_modelo]

    # 4. Predicción
    prediccion = float(model.predict(df_procesado)[0])

    st.success(f"### Nota Final Estimada: {prediccion:.3f}")
    if prediccion >= 3.0:
        st.info("El estudiante probablemente **aprobará** el curso.")
    else:
        st.info("El estudiante probablemente **no aprobará** el curso.")

    with st.expander("Ver variables procesadas enviadas al modelo"):
        st.dataframe(df_procesado)
