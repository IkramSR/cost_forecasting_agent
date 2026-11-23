# Sistema de IA para proyección de costos y planeación presupuestal

Solución analítica para estimar y proyectar el costo operativo de dos equipos críticos en un proyecto de construcción a partir de series históricas de materias primas (`X`, `Y`, `Z`), expuesta mediante una aplicación interactiva en Streamlit con un agente conversacional y desplegada en Azure.

**App en vivo (Azure Container Apps):** `https://cost-forecasting-app.purpleocean-f9e00439.centralus.azurecontainerapps.io/`

## Resumen de la solución

* **Modelado:** Enfoque estructural en dos etapas. Primero se estimaron las relaciones de largo plazo entre insumos y equipos mediante OLS sin intercepto (verificando estacionariedad con ADF y residuos ruido blanco), descubriendo que el Equipo 1 depende de `Y` (~80%) y `X` (~20%), mientras que el Equipo 2 se distribuye de forma equitativa (~33.3%) entre `X`, `Y` y `Z`. Luego se proyectaron los meses faltantes de insumos con `auto_arima`, aprovechando los datos reales adelantados de los archivos individuales (`X.csv` e `Y.csv`) y calculando intervalos al 95% por propagación de varianza.
* **Validación:** Backtesting *Rolling Origin* (`TimeSeriesSplit`, 4 ventanas de 3 meses evitandi fuga de datos), obteniendo un MAPE promedio *ex-ante* de 8.73% (Equipo 1) y 4.86% (Equipo 2), y un error a posteriori menor al 1% útil para auditar cotizaciones de proveedores.
* **Agente de IA:** Construido con LangChain y Gemini API sobre Streamlit. Cuenta con tres herramientas: consulta de pronósticos y métricas vigentes desde Azure Blob Storage (`gold/`), simulador de escenarios porcentuales sobre materias primas y búsqueda web en vivo para contexto de mercado.
* **Mantenibilidad:** El pipeline batch (`src/train_pipeline.py`) actualiza `forecast_resultados.csv` y `model_metadata.json` en Azure Blob Storage para que la aplicación consuma nuevos pesos y proyecciones sin modificar código ni redesplegar el contenedor.

---

## Arquitectura en Azure

<img width="1920" height="539" alt="Diagrama_infraestructura" src="https://github.com/user-attachments/assets/7ae3da57-5a1b-4bb2-8102-769659233f10" />

* **Almacenamiento:** Azure Storage Account (esquema Medallón: `bronze/` para CSVs crudos y `gold/` para proyecciones y metadatos).
* **Despliegue:** Contenedor Docker compilado en Azure Container Registry y expuesto en **Azure Container Apps** (escala a 0 réplicas sin tráfico).


## Estructura del repositorio

```text
cost_forecasting_agent/
├── app/
│   ├── agent.py              # Configuración del agente LangChain y sus herramientas
│   ├── storage.py            # Lectura desde Azure Blob Storage (con respaldo local)
│   ├── streamlit_app.py      # Interfaz de dashboard y chat
│   └── requirements.txt      # Dependencias livianas para el contenedor
├── architecture/
│   └── arquitectura_azure.png
├── data/
│   ├── raw/                  # Datos históricos originales
│   └── processed/            # forecast_resultados.csv y model_metadata.json
├── notebooks/
│   ├── 01_eda_limpieza.ipynb
│   └── 02_modeling_and_forecasting.ipynb
├── src/
│   └── train_pipeline.py     # Script batch de entrenamiento y actualización en gold/
├── Dockerfile
├── .dockerignore
└── requirements.txt          # Dependencias completas (notebooks + pipeline + app)
