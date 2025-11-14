import pandas as pd
from fastapi import HTTPException

# ¡CAMBIO! Creamos espacios separados para cada tipo de contexto.
app_state = {
    "df_qualitative_context": None, # Para el Reporte Final (análisis cualitativo)
    "df_quantitative_context": None, # Para los 13G Puntos (datos cuantitativos)
    "thesis_context_text": ""
}

# --- DEPENDENCIAS ---
# Las modificamos para que devuelvan el contexto correcto.
def get_qualitative_context() -> pd.DataFrame:
    if app_state["df_qualitative_context"] is None:
        raise HTTPException(status_code=503, detail="El contexto cualitativo histórico no está cargado.")
    return app_state["df_qualitative_context"]

def get_quantitative_context() -> pd.DataFrame:
    if app_state["df_quantitative_context"] is None:
        raise HTTPException(status_code=503, detail="El contexto cuantitativo histórico no está cargado.")
    return app_state["df_quantitative_context"]

def get_thesis_context() -> str:
    if not app_state.get("thesis_context_text"):
        raise HTTPException(status_code=503, detail="El contexto de la tesis no está cargado.")
    return app_state["thesis_context_text"]