import io
import json
import os
from pathlib import Path
import pandas as pd
from azure.storage.blob import BlobServiceClient

DIR_LOCAL = Path(__file__).resolve().parent.parent / "data" / "processed"

def _leer_archivo(nombre: str) -> bytes:
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn_str:
        cliente = BlobServiceClient.from_connection_string(conn_str)
        blob = cliente.get_blob_client(container="gold", blob=nombre)
        return blob.download_blob().readall()
    return (DIR_LOCAL / nombre).read_bytes()

def cargar_forecast_df() -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_leer_archivo("forecast_resultados.csv")))

def cargar_metadata() -> dict:
    return json.loads(_leer_archivo("model_metadata.json").decode("utf-8"))

def fuente_datos() -> str:
    return "Azure Blob Storage capa gold" if os.getenv("AZURE_STORAGE_CONNECTION_STRING") else "Cargado desde local"