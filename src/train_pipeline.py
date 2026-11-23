import io
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
import statsmodels.api as sm
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESSED = RAIZ / "data" / "processed"
HORIZONTE = int(os.getenv("FORECAST_HORIZON", "6"))

HISTORICO_EQ = "historico_equipos.csv"
MATERIA_X = "X.csv"
MATERIA_Y = "Y.csv"
MATERIA_Z = "Z.csv"
FEATURES = ["Price_X", "Price_Y", "Price_Z"]


def leer_desde_bronze(nombre_archivo: str) -> bytes:
    """Lee los datos sin procesar de los históricos de equipos y materias primas desde el contenedor "bronze" en Azure Blob Storage, o bien local en caso de fallo de conexión """
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn_str:
        cliente = BlobServiceClient.from_connection_string(conn_str)
        blob = cliente.get_blob_client(container="bronze", blob=nombre_archivo)
        return blob.download_blob().readall()
    return (DIR_RAW / nombre_archivo).read_bytes()


def guardar_en_gold(nombre_archivo: str, contenido: bytes) -> None:
    """Guarda los datos procesados en local y los sube al contenedor 'gold' en Azure Blob Storage."""
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    (DIR_PROCESSED / nombre_archivo).write_bytes(contenido)

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn_str:
        cliente = BlobServiceClient.from_connection_string(conn_str)
        blob = cliente.get_blob_client(container="gold", blob=nombre_archivo)
        blob.upload_blob(contenido, overwrite=True)
        print(f"Subido a Azure Blob Storage (gold/{nombre_archivo})")


def predecir_huecos(serie_hist: pd.Series, serie_fut: pd.Series):
    """Rellena meses faltantes de un insumo con auto_arima y devuelve (valores_completos, sigma)."""
    combinada = pd.concat([serie_hist, serie_fut.dropna()])
    faltantes = serie_fut.isna()
    sigma = np.zeros(len(serie_fut))

    if faltantes.any():
        modelo = pm.auto_arima(combinada, seasonal=False, suppress_warnings=True)
        preds, conf = modelo.predict(n_periods=int(faltantes.sum()), return_conf_int=True, alpha=0.05)
        serie_fut = serie_fut.copy()
        serie_fut.loc[faltantes] = preds.values
        sigma[faltantes.values] = (conf[:, 1] - conf[:, 0]) / (2 * 1.96)

    return serie_fut, sigma


def ejecutar_pipeline():
    print("Inicia pipeline de entrenamiento y forecasting")

    #Formateo de datos
    df_master = pd.read_csv(io.BytesIO(leer_desde_bronze(HISTORICO_EQ)))
    df_master["Date"] = pd.to_datetime(df_master["Date"], format="ISO8601")
    df_por_mes = df_master.set_index("Date").resample("ME").mean().dropna()

    df_x = pd.read_csv(io.BytesIO(leer_desde_bronze(MATERIA_X)))
    df_x["Date"] = pd.to_datetime(df_x["Date"], format="mixed")
    x_mensual = df_x.set_index("Date").resample("ME").mean().rename(columns={"Price": "Price_X"})

    df_y = pd.read_csv(io.BytesIO(leer_desde_bronze(MATERIA_Y)), sep=";", decimal=",")
    df_y["Date"] = pd.to_datetime(df_y["Date"], dayfirst=True)
    y_mensual = df_y.set_index("Date").resample("ME").mean().rename(columns={"Price": "Price_Y"})

    # Validación 4 ventanas de 3 meses
    tscv = TimeSeriesSplit(n_splits=4, test_size=3)
    metricas = {
        "Error_Eq1_Modelo_Estructural": [],
        "Error_Eq1_Modelo_Ingenuo": [],
        "Error_Eq2_Modelo_Estructural": [],
        "Error_Eq2_Modelo_Ingenuo": [],
        "Error_Eq1_(conociendo_xyz)": [],
        "Error_Eq2_(conociendo_xyz)": [],
    }

    for train_idx, test_idx in tscv.split(df_por_mes):
        train, test = df_por_mes.iloc[train_idx], df_por_mes.iloc[test_idx]
        m1 = sm.OLS(train["Price_Equipo1"], train[FEATURES]).fit()
        m2 = sm.OLS(train["Price_Equipo2"], train[FEATURES]).fit()

        insumos_pred = pd.DataFrame(index=test.index)
        for col in FEATURES:
            arima = pm.auto_arima(train[col], seasonal=False, suppress_warnings=True)
            insumos_pred[col] = arima.predict(n_periods=len(test)).values

        metricas["Error_Eq1_Modelo_Estructural"].append(mean_absolute_percentage_error(test["Price_Equipo1"], m1.predict(insumos_pred)) * 100)
        metricas["Error_Eq2_Modelo_Estructural"].append(mean_absolute_percentage_error(test["Price_Equipo2"], m2.predict(insumos_pred)) * 100)
        metricas["Error_Eq1_Modelo_Ingenuo"].append(mean_absolute_percentage_error(test["Price_Equipo1"], np.full(len(test), train["Price_Equipo1"].iloc[-1])) * 100)
        metricas["Error_Eq2_Modelo_Ingenuo"].append(mean_absolute_percentage_error(test["Price_Equipo2"], np.full(len(test), train["Price_Equipo2"].iloc[-1])) * 100)
        metricas["Error_Eq1_(conociendo_xyz)"].append(mean_absolute_percentage_error(test["Price_Equipo1"], m1.predict(test[FEATURES])) * 100)
        metricas["Error_Eq2_(conociendo_xyz)"].append(mean_absolute_percentage_error(test["Price_Equipo2"], m2.predict(test[FEATURES])) * 100)

    #Modelo de regresión con 100% de los datosy proyección a futuro
    mod_eq1 = sm.OLS(df_por_mes["Price_Equipo1"], df_por_mes[FEATURES]).fit()
    mod_eq2 = sm.OLS(df_por_mes["Price_Equipo2"], df_por_mes[FEATURES]).fit()

    h_futuro = pd.date_range(start=df_por_mes.index[-1], periods=HORIZONTE + 1, freq="ME")[1:]
    df_proy = pd.DataFrame(index=h_futuro).join(x_mensual[["Price_X"]]).join(y_mensual[["Price_Y"]])
    df_proy["Price_Z"] = np.nan

    df_proy["Price_X"], sig_x = predecir_huecos(df_por_mes["Price_X"], df_proy["Price_X"])
    df_proy["Price_Y"], sig_y = predecir_huecos(df_por_mes["Price_Y"], df_proy["Price_Y"])
    df_proy["Price_Z"], sig_z = predecir_huecos(df_por_mes["Price_Z"], df_proy["Price_Z"])

    df_proy["Forecast_Equipo1"] = mod_eq1.predict(df_proy[FEATURES])
    df_proy["Forecast_Equipo2"] = mod_eq2.predict(df_proy[FEATURES])

    # Propagación de varianza para intervalos al 95%
    p1, p2 = mod_eq1.params, mod_eq2.params
    s1 = np.sqrt((p1["Price_X"] * sig_x) ** 2 + (p1["Price_Y"] * sig_y) ** 2 + (p1["Price_Z"] * sig_z) ** 2 + mod_eq1.mse_resid)
    s2 = np.sqrt((p2["Price_X"] * sig_x) ** 2 + (p2["Price_Y"] * sig_y) ** 2 + (p2["Price_Z"] * sig_z) ** 2 + mod_eq2.mse_resid)

    df_proy["LI_Equipo1"], df_proy["LS_Equipo1"] = df_proy["Forecast_Equipo1"] - 1.96 * s1, df_proy["Forecast_Equipo1"] + 1.96 * s1
    df_proy["LI_Equipo2"], df_proy["LS_Equipo2"] = df_proy["Forecast_Equipo2"] - 1.96 * s2, df_proy["Forecast_Equipo2"] + 1.96 * s2

    #Exportar datos a "gold" (CSV + JSON)
    metadata = {
        "fecha_ultimo_dato": str(df_por_mes.index[-1].date()),
        "horizonte_meses": HORIZONTE,
        "pesos_ols": {
            "Equipo1": {k: round(float(v), 4) for k, v in p1.items()},
            "Equipo2": {k: round(float(v), 4) for k, v in p2.items()},
        },
        "metricas_backtest_mape": {k: round(float(np.mean(v)), 2) for k, v in metricas.items()},
    }

    guardar_en_gold("forecast_resultados.csv", df_proy.to_csv(index_label="Mes_Proyectado").encode("utf-8"))
    guardar_en_gold("model_metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8"))
    print("Pipeline batch completado exitosamente.")


if __name__ == "__main__":
    ejecutar_pipeline()