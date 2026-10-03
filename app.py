import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación permite ingresar datos manualmente o subir un archivo Excel para realizar predicciones utilizando el modelo de Bagging optimizado.")

# Cargar componentes previamente entrenados
try:
    one_hot_columns = joblib.load('one_hot_columns.joblib')
    scaler = joblib.load('min_max_scaler.joblib')
    model = joblib.load('bagging_optimizado.joblib')
except Exception as e:
    st.error(f"Error al cargar los archivos .joblib (asegúrate de que existen en el entorno): {e}")

# Selector del modo de entrada
modo_entrada = st.radio("Selecciona el modo de entrada de datos:", ["Manual", "Subir archivo Excel"])

def procesar_registro(df_raw, columnas_esperadas, scaler_obj):
    # 1. Crear copia de trabajo
    df_proc = df_raw.copy()
    
    # 2. Eliminar variables ID y Año - Semestre si están presentes
    columnas_a_eliminar = ['ID', 'Año - Semestre', 'Nota_final', 'Aprobo']
    df_proc = df_proc.drop(columns=[col for col in columnas_a_eliminar if col in df_proc.columns], errors='ignore')
    
    # 3. Aplicar One-Hot Encoding manual basado en la lista original de columnas del modelo
    if 'Felder' in df_proc.columns:
        for col in columnas_esperadas:
            if col.startswith('Felder_'):
                categoria = col.replace('Felder_', '')
                df_proc[col] = (df_proc['Felder'] == categoria).astype(float)
        df_proc = df_proc.drop(columns=['Felder'], errors='ignore')
    else:
        # Si no existe la columna Felder, inicializamos las columnas One-Hot en 0
        for col in columnas_esperadas:
            if col.startswith('Felder_'):
                df_proc[col] = 0.0
                
    # 4. Normalizar la variable 'Examen_admisión'
    if 'Examen_admisión' in df_proc.columns:
        # Transformación con el min_max_scaler entrenado
        df_proc['Examen_admision_scaled'] = scaler_obj.transform(df_proc[['Examen_admisión']])
        df_proc = df_proc.drop(columns=['Examen_admisión'], errors='ignore')
    else:
        df_proc['Examen_admision_scaled'] = 0.0
        
    # Reordenar y asegurar que existan exactamente las columnas esperadas por el modelo
    columnas_finales = [col for col in columnas_esperadas if col in df_proc.columns]
    df_proc = df_proc[columnas_finales]
    return df_proc

if modo_entrada == "Manual":
    st.header("Entrada de Datos Manual")
    opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
    
    felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder)
    examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01)
    
    if st.button("Realizar Predicción");
        df_manual = pd.DataFrame([{
            'Felder': felder_input,
            'Examen_admisión': examen_input
        }])
        
        df_procesado = procesar_registro(df_manual, one_hot_columns, scaler)
        prediccion = model.predict(df_procesado)
        
        st.write("### Registro Procesado para el Modelo:")
        st.dataframe(df_procesado)
        st.success(f"La predicción del modelo (Nota Final Estimada) es: {prediccion[0]:.4f}")

else:
    st.header("Subir archivo Excel")
    archivo_cargado = st.file_uploader("Selecciona un archivo Excel (.xlsx):", type=["xlsx"])
    
    if archivo_cargado is not None:
        df_excel = pd.read_excel(archivo_cargado)
        st.write("### Vista previa del archivo original:")
        st.dataframe(df_excel.head())
        
        if st.button("Procesar y Predicir Archivo"):
            try:
                # Procesar todos los registros del Excel subido
                df_procesado_batch = procesar_registro(df_excel, one_hot_columns, scaler)
                
                # Realizar predicciones de forma masiva
                predicciones = model.predict(df_procesado_batch)
                
                # Adjuntar predicciones al DataFrame original para fácil lectura
                df_resultado = df_excel.copy()
                df_resultado['Nota_final_estimada'] = predicciones
                
                st.write("### Resultados de las Predicciones:")
                st.dataframe(df_resultado)
                
                # Permitir descargar el resultado procesado en un nuevo Excel
                output_excel = df_resultado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar resultados en CSV",
                    data=output_excel,
                    file_name="predicciones_curso.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Ocurrió un error al procesar el archivo: {e}")
