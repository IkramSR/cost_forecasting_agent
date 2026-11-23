import json
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from ddgs import DDGS
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DIR_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
RUTA_CSV = DIR_PROCESSED / "forecast_resultados.csv"
RUTA_META = DIR_PROCESSED / "model_metadata.json"

def cargar_metadata() -> dict:
    with open(RUTA_META, "r", encoding="utf-8") as f:
        return json.load(f)

def consultar_pronostico() -> str:
    """Devuelve la tabla actual de proyecciones de costos (IC 95%) y los metadatos vigentes del modelo (pesos OLS y métricas MAPE del último entrenamiento)."""
    df = pd.read_csv(RUTA_CSV)
    meta = cargar_metadata()
    return (
        f"=== PROYECCIONES (Último corte: {meta['fecha_ultimo_dato']}) ===\n"
        f"{df.to_string(index=False)}\n\n"
        f"=== PESOS ESTRUCTURALES (OLS) Y MÉTRICAS ===\n"
        f"{json.dumps(meta, indent=2, ensure_ascii=False)}"
    )

def simular_escenario(cambio_x_pct: float = 0.0, cambio_y_pct: float = 0.0, cambio_z_pct: float = 0.0) -> str:
    """Simula cómo cambiaría el costo proyectado del último mes si las materias primas X, Y o Z varían un porcentaje dado"""
    df = pd.read_csv(RUTA_CSV)
    pesos = cargar_metadata()["pesos_ols"]
    ultimo = df.iloc[-1]

    nx = ultimo["Price_X"] * (1 + cambio_x_pct / 100)
    ny = ultimo["Price_Y"] * (1 + cambio_y_pct / 100)
    nz = ultimo["Price_Z"] * (1 + cambio_z_pct / 100)

    p1, p2 = pesos["Equipo1"], pesos["Equipo2"]
    nuevo_eq1 = p1["Price_X"] * nx + p1["Price_Y"] * ny + p1["Price_Z"] * nz
    nuevo_eq2 = p2["Price_X"] * nx + p2["Price_Y"] * ny + p2["Price_Z"] * nz

    d1 = ((nuevo_eq1 - ultimo["Forecast_Equipo1"]) / ultimo["Forecast_Equipo1"]) * 100
    d2 = ((nuevo_eq2 - ultimo["Forecast_Equipo2"]) / ultimo["Forecast_Equipo2"]) * 100

    return (
        f"Simulación para {ultimo['Mes_Proyectado']} (X:{cambio_x_pct:+.1f}%, Y:{cambio_y_pct:+.1f}%, Z:{cambio_z_pct:+.1f}%):\n"
        f"- Nuevo Equipo 1: {nuevo_eq1:.2f} ({d1:+.2f}% vs base {ultimo['Forecast_Equipo1']:.2f})\n"
        f"- Nuevo Equipo 2: {nuevo_eq2:.2f} ({d2:+.2f}% vs base {ultimo['Forecast_Equipo2']:.2f})"
    )

def buscar_mercado_externo(consulta: str) -> str:
    """Busca noticias recientes o tendencias de materias primas y sector construcción."""
    try:
        resultados = DDGS(timeout=10).text(consulta, max_results=3)
        if not resultados:
            return "No se encontraron noticias recientes."
        return "\n".join(f"- {r['title']}: {r['body']} ({r['href']})" for r in resultados)
    except Exception as e:
        return f"Buscador externo no disponible temporalmente: {e}"

PROMPT_SISTEMA = """
Eres un consultor experto en gestión de costos y analítica financiera para un proyecto de construcción.

TEMAS PERMITIDOS (responde siempre que pregunten sobre cualquiera de estos):
- Costos proyectados, intervalos de confianza y fechas de Equipo 1 y Equipo 2.
- Materias primas X, Y, Z, sus pesos estructurales (OLS), backtesting, MAPE y metodología del modelo.
- Simulaciones de escenarios ("¿qué pasa si X, Y o Z suben o bajan un %?").
- Noticias externas, inflación, commodities industriales y tendencias del mercado de la construcción.

INSTRUCCIONES DE USO DE HERRAMIENTAS:
1. Usa `consultar_pronostico` SIEMPRE que pregunten por costos futuros, meses, intervalos, recetas o métricas del modelo. Nunca inventes cifras.
2. Usa `simular_escenario` cuando pregunten por impactos porcentuales hipotéticos en X, Y o Z.
3. Usa `buscar_mercado_externo` cuando pidan noticias, tendencias o contexto de mercado. IMPORTANTE: como las materias primas X, Y y Z están anonimizadas por confidencialidad en el dataset, busca tendencias de costos de materiales de construcción y commodities industriales (ej. acero, aluminio, energía, inflación en construcción) y aclara brevemente que usas esos referentes de mercado para complementar nuestro pronóstico.
4. Responde en español claro, ejecutivo y conciso.

TEMAS FUERA DE ALCANCE:
Únicamente si el usuario pregunta sobre temas completamente ajenos al negocio (por ejemplo: recetas de cocina, deportes, entretenimiento, chistes, política ajena a economía/construcción o programación general), responde amablemente que solo estás autorizado para asistir en el análisis de costos del proyecto y el mercado de construcción.
"""

def construir_agente():
    modelo = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    llm = ChatGoogleGenerativeAI(
        model=modelo,
        temperature=0.2,
        timeout=30,
        max_retries=1,   
    )
    return create_agent(
        model=llm,
        tools=[consultar_pronostico, simular_escenario, buscar_mercado_externo],
        system_prompt=PROMPT_SISTEMA,
    )

def extraer_texto(contenido) -> str:
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(b.get("text", "") for b in contenido if isinstance(b, dict))
    return str(contenido)