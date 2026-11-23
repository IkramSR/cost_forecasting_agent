from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from agente import construir_agente, extraer_texto

st.set_page_config(page_title="Sistema de IA para proyección de costos y planeación presupuestal", layout="wide")
st.title("Agente conversacional sobre Proyección de Costos")
st.caption("Herramienta de análisis y estimación de costos de equipos según las variaciones de materias primas necesarias para su manufactura, apoyando la planeación presupuestal en horizontes mensuales")

RUTA_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "forecast_resultados.csv"
df_forecast = pd.read_csv(RUTA_CSV)

# No se recarga el cache en cada mensaje
@st.cache_resource
def obtener_agente():
    return construir_agente()

tab1, tab2 = st.tabs(["Proyección de Costos", "Conversa con el Agente de IA"])

# Pestaña 1, dashboard sencillo
with tab1:
    st.subheader("Proyecciones a 6 meses")
    ultimo = df_forecast.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Equipo 1 (Último mes estimado)", f"${ultimo['Forecast_Equipo1']:,.2f}",
        f"Rango esperado: [{ultimo['LI_Equipo1']:,.1f} - {ultimo['LS_Equipo1']:,.1f}]"
    )

    col2.metric(
        "Equipo 2 (Último mes estimado)", f"${ultimo['Forecast_Equipo2']:,.2f}",
        f"Rango esperado: [{ultimo['LI_Equipo2']:,.1f} - {ultimo['LS_Equipo2']:,.1f}]"
    )

    col3.metric(
        "Error promedio (MAPE 3m)",
        "Eq1: 8.73% | Eq2: 4.86%",
        "Último error observado: 0.29%",
        delta_color="blue"
    )

    # Gráfica con intérvalos de confianza
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    fechas = df_forecast["Mes_Proyectado"]

    # Equipo 1
    ax1.plot(fechas, df_forecast["Forecast_Equipo1"], marker="o", color="#365F7D", label="Proyección")
    ax1.fill_between(fechas, df_forecast["LI_Equipo1"], df_forecast["LS_Equipo1"],
        color="#365F7D", alpha=0.15, label="Rango esperado (95%)")
    ax1.set_title("Equipo 1 (80% Y + 20% X)")
    ax1.tick_params(axis="x", rotation=20)
    ax1.legend(frameon=False)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Equipo 2
    ax2.plot(fechas, df_forecast["Forecast_Equipo2"], marker="o", color="#648B85", label="Proyección")
    ax2.fill_between(
        fechas, df_forecast["LI_Equipo2"], df_forecast["LS_Equipo2"],
        color="#648B85", alpha=0.15, label="Rango esperado (95%)")
    ax2.set_title("Equipo 2 (33% X + 33% Y + 33% Z)")
    ax2.tick_params(axis="x", rotation=20)
    ax2.legend(frameon=False)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Segunda pestaña, del agente de IA
with tab2:
    st.subheader("Conversa con el Agente sobre el Pronóstico y el Mercado")
    st.caption("Ejemplos que puedes preguntar: *'¿Cuál es el costo esperado del Equipo 1 y su rango de riesgo?'*, "
               "*'¿Qué pasa si la materia prima Y sube un 15%?'*, o "
               "*'¿Cuál será el costo del equipo 2 en 4 meses y cómo podría variar según los costos del mercado?'*.")

    # Memoria del agente en la sesión de streamlit
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    # mostrar mensajes anteriores
    for msg in st.session_state.mensajes:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    #Recuadros de texto
    if pregunta := st.chat_input("Escribe tu pregunta para el agente..."):
        st.session_state.mensajes.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Consultando modelo y mercado..."):
                try:
                    agente = obtener_agente()
                    respuesta = agente.invoke(
                        {"messages": st.session_state.mensajes},
                        config={"recursion_limit": 8},
                    )
                    texto_final = extraer_texto(respuesta["messages"][-1].content)
                except Exception as e:
                    texto_final = f" Ocurrió un error al consultar el modelo: {e}"
                st.markdown(texto_final)

        st.session_state.mensajes.append({"role": "assistant", "content": texto_final})