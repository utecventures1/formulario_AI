# main.py
import pandas as pd
import io
import json
import os
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import time

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

if not API_KEY:
    print("ERROR: La variable de entorno GOOGLE_API_KEY no se encontró.")
    print("Asegúrate de que tu archivo .env está en la carpeta 'backend' y tiene el formato correcto.")
else:
    print(f"API Key cargada exitosamente. Comienza con: {API_KEY[:4]}...")

# Cargamos nuestra configuración de scoring desde el archivo JSON
with open("scoring_config.json", "r") as f:
    config_data = json.load(f)
    # Le decimos al programa que use el diccionario que está DENTRO de "SCORING_CATEGORIES"
    SCORING_CONFIG = config_data["SCORING_CATEGORIES"]

FEEDBACK_LOG_FILE = "feedback_log.csv"

app = FastAPI(
    title="Deal Flow AI API v2",
    description="Sube un archivo de postulaciones y un archivo de portafolio para obtener un scoring contextualizado."
)

# --- Configuración de CORS ---
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Funciones de Lógica de Negocio ---
def analyze_portfolio(df_portfolio: pd.DataFrame) -> str:
    """Analiza el portafolio y devuelve un resumen en texto para el prompt."""
    print("Analizando el portafolio para extraer contexto...")
    industry_col = 'Sector'
    if industry_col in df_portfolio.columns:
        top_industries = df_portfolio[industry_col].value_counts().head(3).index.tolist()
        return f"Las industrias clave son: {', '.join(top_industries)}."
    return "No se pudo determinar un patrón de industrias en el portafolio."

def get_feedback_examples() -> str:
    """Lee el log de feedback y formatea algunos ejemplos para el prompt."""
    try:
        if os.path.exists(FEEDBACK_LOG_FILE):
            df_feedback = pd.read_csv(FEEDBACK_LOG_FILE)
            # Tomamos los 2 ejemplos más recientes y de alta calidad
            examples = df_feedback.tail(2).to_dict('records')
            return json.dumps(examples, indent=2)
    except Exception:
        return "No hay ejemplos de feedback disponibles."
    return "No hay ejemplos de feedback disponibles."

async def extract_text_from_pdf(pdf_file) -> str:
    """Lee el contenido de un archivo PDF y devuelve su texto."""
    try:
        pdf_content = await pdf_file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"Error al leer el PDF: {e}")
        return "No se pudo leer el documento de contexto."

def get_llm_dimensional_scoring(startup_data: str, feedback_examples: str, thesis_context: str, portfolio_context: str) -> dict:

    if not API_KEY:
         # Si la API Key no se cargó al inicio, devolvemos un error claro.
        return {"dimensional_scores": {category: 0 for category in SCORING_CONFIG}, "error": "API Key no configurada."}

    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    scoring_instructions = "\n".join(
        f'- "{category}": {details["descripcion_prompt"]}'
        for category, details in SCORING_CONFIG.items()
    )

    prompt = f"""
    Eres un analista experto de Venture Capital. Tu tarea es analizar una startup y puntuarla en varias dimensiones.

    **CONTEXTO FUNDAMENTAL - Tesis de Inversión de UTEC Ventures:**
    {thesis_context}

    **Ejemplos de análisis previos corregidos por analistas senior (para tu referencia):**
    {feedback_examples}

    **Datos de la Startup Postulante:**
    {startup_data}

    **Instrucciones:**
    Basándote en la Tesis de Inversión y los ejemplos, evalúa la startup en las siguientes dimensiones.
    Responde ÚNICAMENTE con un objeto JSON válido con una clave "dimensional_scores" que contenga un objeto con las siguientes claves y un puntaje de 0 a 100 para cada una:
    {scoring_instructions}
    
    Ejemplo de formato de salida:
    {{
      "dimensional_scores": {{
        "equipo": 85,
        "producto": 70,
        "tesis_utec": 90,
        "oportunidad": 75,
        "validacion": 50
      }}
    }}
    """
    
    try:
        print("    -> Enviando prompt a la IA. Esperando respuesta...")
        response = model.generate_content(prompt)
        print("    -> Respuesta recibida de la IA. Intentando procesar...")
        print(f"    -> TEXTO CRUDO RECIBIDO: {response.text}")

        json_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_response)
    except Exception as e:
        # --- PRINT DE DEPURACIÓN 3: ¡VER EL ERROR REAL! ---
        print(f"\n    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"    !!! ERROR AL LLAMAR O PROCESAR LA RESPUESTA DE LA IA !!!")
        print(f"    !!! ERROR DETALLADO: {e}")
        print(f"    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        
        # Devolvemos el diccionario de ceros para que el programa no se caiga
        return {"dimensional_scores": {category: 0 for category in SCORING_CONFIG}}

# --- ENDPOINTS DE LA API ---

@app.post("/api/process-and-score")
async def process_and_score(
    applications_file: UploadFile = File(...),
    portfolio_file: UploadFile = File(...),
    context_pdf: Optional[UploadFile] = File(None)
):
    apps_content = await applications_file.read()
    df_apps = pd.read_csv(io.BytesIO(apps_content))
    
    portfolio_content = await portfolio_file.read()
    df_portfolio = pd.read_csv(io.BytesIO(portfolio_content))
    portfolio_context = analyze_portfolio(df_portfolio)

    thesis_context = "No se proporcionó un documento de tesis."
    if context_pdf:
        # --- CAMBIO 2: Ahora 'await' la llamada a la función asíncrona ---
        thesis_context = await extract_text_from_pdf(context_pdf)

    feedback_examples = get_feedback_examples()
    results = []
    
    total_startups = len(df_apps)
    print(f"\n--- INICIANDO PROCESO DE SCORING PARA {total_startups} STARTUPS ---")

    for index, row in df_apps.iterrows():
        
        # --- CAMBIO 3: Imprimimos el progreso por cada fila ---
        startup_name = row.get('Nombre de la startup', f'Fila {index + 1}')
        print(f"\n[ {index + 1} / {total_startups} ] Procesando: '{startup_name}'...")
        
        startup_data_str = row.to_json()
        
        llm_result = get_llm_dimensional_scoring(startup_data_str, feedback_examples, thesis_context, portfolio_context)
        dimensional_scores = llm_result.get("dimensional_scores", {})
        
        final_score = 0
        for category, details in SCORING_CONFIG.items():
            # Ahora dimensional_scores es el diccionario correcto: {"equipo": 65, ...}
            score = dimensional_scores.get(category, 0)
            final_score += score * details["peso"]

        result_row = row.where(pd.notna(row), None).to_dict()

        result_row["ai_scores"] = dimensional_scores
        result_row["final_weighted_score"] = round(final_score, 2)
        results.append(result_row)
        
        print(f" -> Análisis completado para '{startup_name}'. Score final: {result_row['final_weighted_score']}")

        print("    -> Esperando 1.1 segundos para no exceder la cuota...")

    print("\n--- PROCESO DE SCORING FINALIZADO ---")
    return results

@app.post("/api/submit-feedback")
async def submit_feedback(data: dict = Body(...)):
    """
    Recibe el feedback del analista humano y lo guarda en el log.
    El 'data' debería contener los datos de la startup y los puntajes corregidos por el humano.
    """
    try:
        df_feedback = pd.DataFrame([data])
        # Guardamos en el CSV. Si el archivo no existe, lo crea. Si existe, añade la fila.
        df_feedback.to_csv(FEEDBACK_LOG_FILE, mode='a', header=not os.path.exists(FEEDBACK_LOG_FILE), index=False)
        return {"status": "Feedback guardado con éxito."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo guardar el feedback: {e}")
