import os
import pandas as pd
import fitz
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Importamos los routers de la API
from api.analysis import router as analysis_router
from api.config import router as config_router

# Importamos nuestro contenedor de estado
from dependencies import app_state

# --- CONFIGURACIÓN INICIAL DE LA APP ---
load_dotenv()
app = FastAPI(
    title="Deal Flow AI API v2",
    description="API para puntuar postulaciones de startups y analizar datos históricos."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- ¡NUEVAS RUTAS A ARCHIVOS DE CONTEXTO! ---
PUNTOS_CSV_PATH = "13G_puntos.csv"
HISTORICOS_CSV_PATH = "Reporte_Final_con_Historicos.csv" # ¡Corregido!
CONTEXT_PDF_PATH = "contexto_uv.pdf"


# --- EVENTO DE INICIO (STARTUP) ---
@app.on_event("startup")
async def startup_event():
    print("--- 🚀 Iniciando la aplicación y cargando datos de contexto... ---")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❗️ ERROR CRÍTICO: GOOGLE_API_KEY no encontrada.")
    else:
        genai.configure(api_key=api_key)
        print(f"✅ API Key de Google cargada: {api_key[:4]}...")

    try:
        # --- LÓGICA DE CARGA SEPARADA ---
        print("1. Cargando archivos de datos de contexto...")
        df_historicos = pd.read_csv(HISTORICOS_CSV_PATH)
        df_puntos = pd.read_csv(PUNTOS_CSV_PATH)
        
        # Guardamos cada DataFrame en su lugar correspondiente en el estado.
        app_state["df_qualitative_context"] = df_historicos
        app_state["df_quantitative_context"] = df_puntos
        
        print(f"  -> Contexto Cualitativo ('{HISTORICOS_CSV_PATH}') cargado ({len(df_historicos)} filas).")
        print(f"  -> Contexto Cuantitativo ('{PUNTOS_CSV_PATH}') cargado ({len(df_puntos)} filas).")
        
        # Cargamos el PDF de contexto.
        print("3. Cargando PDF de contexto...")
        with open(CONTEXT_PDF_PATH, "rb") as pdf_file:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            thesis_text = "".join(page.get_text() for page in doc)
            app_state["thesis_context_text"] = thesis_text
        print(f"✅ PDF de contexto cargado. {len(thesis_text)} caracteres.")
        
        print("\n--- ✅ Carga de contexto finalizada. La API está lista. ---")
    except FileNotFoundError as e:
        print(f"❗️ ERROR CRÍTICO: No se encontró un archivo de contexto: {e}.")
    except Exception as e:
        print(f"❗️ ERROR CRÍTICO al cargar o fusionar datos de contexto: {e}.")


# --- INCLUSIÓN DE RUTAS Y ARCHIVOS ESTÁTICOS ---
app.include_router(analysis_router)
app.include_router(config_router)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")