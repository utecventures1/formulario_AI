import pandas as pd
import json
import asyncio
import re
import time
from typing import List, Dict, Tuple
import google.generativeai as genai

# --- CONFIGURACIÓN Y CONSTANTES ---

# Carga de configuración de puntajes
try:
    with open("scoring_config.json", "r") as f:
        config_data = json.load(f)
    SCORING_CONFIG = config_data.get("SCORING_CATEGORIES", {})
except (FileNotFoundError, json.JSONDecodeError):
    SCORING_CONFIG = {}

# Jerarquía de estados para contexto
STATUS_HIERARCHY = {
    "Investment": {"score": 10, "description": "Éxito máximo. La startup recibió inversión."},
    "Investment committee": {"score": 9, "description": "Etapa final. Muy prometedora, pasó a comité de inversión."},
    "Interview UV": {"score": 8, "description": "Etapa avanzada. Pasó los filtros iniciales y tuvo una entrevista formal."},
    "Reference Checks": {"score": 7, "description": "Etapa de validación. Se están verificando referencias."},
    "Reviewing": {"score": 5, "description": "Etapa media. Superó el screening inicial y está en revisión activa."},
    "Screening": {"score": 4, "description": "Etapa temprana. Apenas se está evaluando si cumple lo mínimo."},
    "Backlog": {"score": 3, "description": "En espera. No es una prioridad ahora."},
    "Rechazo con feed": {"score": 1, "description": "Rechazada. No cumple los criterios, aunque se dio feedback."},
}

# --- CONFIGURACIÓN DE MODELOS Y TIEMPOS DE ESPERA ---
# Estructura: (Nombre del Modelo, Segundos de Espera si tiene éxito)
MODEL_PRIORITY_CONFIG = [
    ("gemini-2.5-flash-lite", 6),   # Prioridad 1: Muy rápido. Espera 6s.
    ("gemini-2.5-flash", 12)        # Prioridad 2: Respaldo potente. Espera 12s.
]

# --- LÓGICA DE SCORING CON IA (CON FALLBACK Y TIEMPOS DINÁMICOS) ---

def get_llm_dimensional_scoring(
    startup_data: str, 
    qualitative_context_json: str, 
    quantitative_context_json: str, 
    thesis_context: str
) -> Tuple[dict, int]: # Retorna: (JSON Resultado, Segundos a esperar)
    
    # Respuesta por defecto en caso de error total
    default_response = {
        "dimensional_scores": {category: 0 for category in SCORING_CONFIG.keys()},
        "qualitative_analysis": {k: "Error" for k in ["project_thesis", "problem", "solution", "key_metrics", "founding_team", "market_and_competition"]},
        "score_justification": {k: "Error" for k in ["equipo", "producto", "tesis_utec", "oportunidad", "validacion"]}
    }

    status_hierarchy_prompt = "\n".join(f'- {s} (Nivel {d["score"]}/10): {d["description"]}' for s, d in STATUS_HIERARCHY.items())
    
    # Construcción del Prompt con Doble Contexto
    prompt = f"""
        Eres un analista de Venture Capital de clase mundial en UTEC Ventures. Tu tarea es analizar una startup candidata usando dos tipos de contexto histórico.

        **CONTEXTO ESTRATÉGICO Y DATOS HISTÓRICOS:**

        1.  **Tesis de Inversión (Nuestra Filosofía):**
            ```
            {thesis_context}
            ```

        2.  **Contexto Histórico CUALITATIVO:**
            Esta es una lista de informes cualitativos de startups que hemos analizado. Úsala para entender nuestro estilo de evaluación.
            ```json
            {qualitative_context_json}
            ```

        3.  **Contexto Histórico CUANTITATIVO:**
            Esta es una tabla con los puntajes numéricos y decisiones finales pasadas. Úsala para calibrar tus puntajes.
            ```json
            {quantitative_context_json}
            ```
        
        **TAREA:**
        Analiza la siguiente startup candidata basándote en TODO el contexto:
        
        **Datos de la Startup:**
        ```json
        {startup_data}
        ```

        **INSTRUCCIONES:**
        Completa la estructura JSON. Tus análisis cualitativos deben reflejar el estilo del contexto CUALITATIVO. Tus puntajes numéricos deben ser consistentes con la escala vista en el contexto CUANTITATIVO.
        
        **Formato de Salida JSON (OBLIGATORIO):**
        ```json
        {{
            "dimensional_scores": {{
                "equipo": <0-100>, "producto": <0-100>, "tesis_utec": <0-100>, 
                "oportunidad": <0-100>, "validacion": <0-100>
            }},
            "qualitative_analysis": {{
                "project_thesis": "Resume la tesis principal.",
                "problem": "Describe el problema.",
                "solution": "Describe la solución.",
                "key_metrics": "Lista las métricas clave.",
                "founding_team": "Describe al equipo fundador.",
                "market_and_competition": "Resume mercado y competencia."
            }},
            "score_justification": {{
                "equipo": "Justifica puntaje equipo.",
                "producto": "Justifica puntaje producto.",
                "tesis_utec": "Justifica puntaje tesis.",
                "oportunidad": "Justifica puntaje oportunidad.",
                "validacion": "Justifica puntaje validacion."
            }}
        }}
        ```
        """

    # Bucle de intentos por modelo (Lite -> Flash)
    for model_name, wait_seconds in MODEL_PRIORITY_CONFIG:
        try:
            print(f" -> Intentando análisis con modelo: {model_name}...")
            model = genai.GenerativeModel(model_name)
            
            # Generar contenido
            response = model.generate_content(prompt)
            
            # Procesar respuesta
            text_response = response.text.strip()
            json_start = text_response.find('{')
            json_end = text_response.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_text = text_response[json_start:json_end]
                # Limpieza básica de errores comunes de JSON en LLMs
                json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
                try:
                    # ÉXITO: Retornamos el JSON y el tiempo de espera asociado al modelo
                    return json.loads(json_text), wait_seconds
                except json.JSONDecodeError:
                    print(f" -> Error de sintaxis JSON en {model_name}.")
                    return default_response, 5 # Error de parseo, espera corta
            else:
                print(f" -> {model_name} no devolvió un JSON válido.")
                return default_response, 5

        except Exception as e:
            error_msg = str(e).lower()
            # Manejo de errores de Cuota (429)
            if "429" in error_msg or "quota" in error_msg or "resource exhausted" in error_msg:
                print(f" ⚠️ Cuota excedida en {model_name}. Cambiando al siguiente modelo...")
                continue # Salta al siguiente modelo en la lista
            elif "not found" in error_msg:
                print(f" ⚠️ Modelo {model_name} no encontrado. Saltando...")
                continue
            else:
                print(f" !!! ERROR CRÍTICO en {model_name}: {e} !!!")
                return default_response, 10

    # Si sale del bucle, fallaron todos
    print(" ❌ SE AGOTARON TODOS LOS MODELOS DISPONIBLES (Cuota o Error).")
    return default_response, 60


# --- BUCLE DE STREAMING (CON PAUSAS DINÁMICAS) ---

async def run_scoring_loop_stream(
    df_to_score: pd.DataFrame, 
    df_qual_context: pd.DataFrame, 
    df_quant_context: pd.DataFrame, 
    thesis_context: str
):
    # Convertir DataFrames a JSON una sola vez
    qual_context_json = df_qual_context.to_json(orient='records', indent=2)
    quant_context_json = df_quant_context.to_json(orient='records', indent=2)

    for index, row in df_to_score.iterrows():
        startup_name = row.get('Nombre de la startup') or row.get('Nombre', f'Fila {index + 1}')
        print(f"\n[ Stream / {index + 1} de {len(df_to_score)} ] Procesando: '{startup_name}'...")
        startup_json = row.where(pd.notna(row), None).to_json()
        
        # Obtenemos RESULTADO y TIEMPO DE ESPERA dinámico
        llm_result, dynamic_wait = get_llm_dimensional_scoring(
            startup_data=startup_json,
            qualitative_context_json=qual_context_json,
            quantitative_context_json=quant_context_json,
            thesis_context=thesis_context
        )
        
        # Calcular puntaje ponderado
        dimensional_scores = llm_result.get("dimensional_scores", {})
        final_score = 0
        if SCORING_CONFIG:
            final_score = sum(
                (dimensional_scores.get(cat, 0) or 0) * details.get("peso", 0)
                for cat, details in SCORING_CONFIG.items()
            )
        
        # Preparar y enviar respuesta
        original_data = row.where(pd.notna(row), None).to_dict()
        result_row = {**original_data, **llm_result, "final_weighted_score": round(final_score, 2)}
        
        yield f"data: {json.dumps(result_row)}\n\n"
        
        # PAUSA DINÁMICA: 6s si fue Lite, 12s si fue Flash
        print(f" -> Stream enviado para '{startup_name}'. Esperando {dynamic_wait}s (según modelo usado)...")
        await asyncio.sleep(dynamic_wait)


# --- FUNCIÓN DE RE-ANÁLISIS (EJECUCIÓN ÚNICA) ---

async def run_single_scoring(
    startup_dict: dict, 
    df_qual_context: pd.DataFrame,
    df_quant_context: pd.DataFrame,
    thesis_context: str
) -> Dict:
    startup_name = startup_dict.get('Nombre de la startup') or startup_dict.get('Nombre', 'Startup sin nombre')
    print(f"\n[ Re-análisis ] Procesando: '{startup_name}'...")

    qual_context_json = df_qual_context.to_json(orient='records', indent=2)
    quant_context_json = df_quant_context.to_json(orient='records', indent=2)
    
    # Ignoramos el tiempo de espera (_) porque es una ejecución única
    llm_result, _ = get_llm_dimensional_scoring(
        startup_data=json.dumps(startup_dict),
        qualitative_context_json=qual_context_json,
        quantitative_context_json=quant_context_json,
        thesis_context=thesis_context
    )
    
    dimensional_scores = llm_result.get("dimensional_scores", {})
    final_score = 0
    if SCORING_CONFIG:
        final_score = sum(
            (dimensional_scores.get(cat, 0) or 0) * details.get("peso", 0)
            for cat, details in SCORING_CONFIG.items()
        )
    
    result_row = {**startup_dict, **llm_result, "final_weighted_score": round(final_score, 2)}
    
    print(f" -> Re-análisis completo para '{startup_name}'. Nuevo Score: {result_row['final_weighted_score']}")
    return result_row
