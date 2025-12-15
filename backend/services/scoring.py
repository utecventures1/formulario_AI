import pandas as pd
import json
import asyncio
import re
from typing import List, Dict
import google.generativeai as genai

# --- CONFIGURACIÓN Y CONSTANTES (Sin cambios) ---
try:
    with open("scoring_config.json", "r") as f:
        config_data = json.load(f)
    SCORING_CONFIG = config_data.get("SCORING_CATEGORIES", {})
except (FileNotFoundError, json.JSONDecodeError):
    SCORING_CONFIG = {}

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


# --- LÓGICA DE SCORING CON IA (PROMPT ACTUALIZADO CON DOBLE CONTEXTO) ---

def get_llm_dimensional_scoring(
    startup_data: str, 
    qualitative_context_json: str, # Contexto 1
    quantitative_context_json: str, # Contexto 2
    thesis_context: str
) -> dict:
    default_response = {
        "dimensional_scores": {category: 0 for category in SCORING_CONFIG.keys()},
        "qualitative_analysis": {k: "Error" for k in ["project_thesis", "problem", "solution", "key_metrics", "founding_team", "market_and_competition"]},
        "score_justification": {k: "Error" for k in ["equipo", "producto", "tesis_utec", "oportunidad", "validacion"]}
    }

    try:
        model = genai.GenerativeModel('gemma-3-27b-it')
        status_hierarchy_prompt = "\n".join(f'- {s} (Nivel {d["score"]}/10): {d["description"]}' for s, d in STATUS_HIERARCHY.items())
        
        prompt = f"""
            Eres un analista de Venture Capital de clase mundial en UTEC Ventures. Tu tarea es analizar una startup candidata usando dos tipos de contexto histórico.

            **CONTEXTO ESTRATÉGICO Y DATOS HISTÓRICOS:**

            1.  **Tesis de Inversión (Nuestra Filosofía):**
                ```
                {thesis_context}
                ```

            2.  **Contexto Histórico CUALITATIVO (Análisis detallados y notas):**
                Esta es una lista de informes cualitativos de startups que hemos analizado. Úsala para entender nuestro estilo de evaluación, razonamiento y el tipo de preguntas que hacemos.
                ```json
                {qualitative_context_json}
                ```

            3.  **Contexto Histórico CUANTITATIVO (Tabla de Puntajes y Decisiones):**
                Esta es una tabla con los puntajes numéricos y decisiones finales que hemos tomado en el pasado. Úsala para calibrar tus puntajes numéricos y entender la correlación entre puntajes y éxito.
                ```json
                {quantitative_context_json}
                ```
            
            **TAREA:**
            Ahora, analiza la siguiente startup candidata basándote en TODO el contexto proporcionado:
            
            **Datos de la Startup a Analizar:**
            ```json
            {startup_data}
            ```

            **INSTRUCCIONES:**
            Completa la estructura JSON. Tus análisis cualitativos deben reflejar el estilo del contexto CUALITATIVO. Tus puntajes numéricos deben ser consistentes con la escala vista en el contexto CUANTITATIVO.
            
            **Formato de Salida JSON (OBLIGATORIO Y ÚNICO):**
            ```json
            {{
                "dimensional_scores": {{
                    "equipo": <0-100>, "producto": <0-100>, "tesis_utec": <0-100>, 
                    "oportunidad": <0-100>, "validacion": <0-100>
                }},
                "qualitative_analysis": {{
                    "project_thesis": "Resume la tesis principal de la startup.",
                    "problem": "Describe el problema.",
                    "solution": "Describe la solución.",
                    "key_metrics": "Lista las métricas clave.",
                    "founding_team": "Describe al equipo fundador.",
                    "market_and_competition": "Resume el mercado y competencia."
                }},
                "score_justification": {{
                    "equipo": "Justifica el puntaje de 'equipo'.",
                    "producto": "Justifica el puntaje de 'producto'.",
                    "tesis_utec": "Justifica el puntaje de 'tesis_utec' alineado a la Tesis de Inversión.",
                    "oportunidad": "Justifica el puntaje de 'oportunidad' (Market Cap, etc.).",
                    "validacion": "Justifica el puntaje de 'validacion' (Logros, tracción)."
                }}
            }}
            ```
            """
        
        print(" -> Enviando prompt con doble contexto a la IA...")
        response = model.generate_content(prompt)
        
        text_response = response.text.strip()
        json_start = text_response.find('{')
        json_end = text_response.rfind('}') + 1

        if json_start != -1 and json_end != -1:
            json_text = text_response[json_start:json_end]
            json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f" -> ERROR: El JSON sigue siendo inválido después de la limpieza: {e}")
                return default_response
        else:
            print(" -> ADVERTENCIA: No se encontró un JSON válido en la respuesta de la IA.")
            return default_response

    except Exception as e:
        print(f"\n !!! ERROR AL PROCESAR RESPUESTA DE LA IA: {e} !!!")
        return default_response


# --- BUCLE DE STREAMING (MODIFICADO PARA DOBLE CONTEXTO) ---

async def run_scoring_loop_stream(
    df_to_score: pd.DataFrame, 
    df_qual_context: pd.DataFrame, 
    df_quant_context: pd.DataFrame, 
    thesis_context: str
):
    qual_context_json = df_qual_context.to_json(orient='records', indent=2)
    quant_context_json = df_quant_context.to_json(orient='records', indent=2)

    for index, row in df_to_score.iterrows():
        startup_name = row.get('Nombre de la startup') or row.get('Nombre', f'Fila {index + 1}')
        print(f"\n[ Stream / {index + 1} de {len(df_to_score)} ] Procesando: '{startup_name}'...")
        startup_json = row.where(pd.notna(row), None).to_json()
        
        llm_result = get_llm_dimensional_scoring(
            startup_data=startup_json,
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
        
        original_data = row.where(pd.notna(row), None).to_dict()
        result_row = {**original_data, **llm_result, "final_weighted_score": round(final_score, 2)}
        
        yield f"data: {json.dumps(result_row)}\n\n"
        
        print(f" -> Stream enviado para '{startup_name}'. Esperando 6s...")
        await asyncio.sleep(10)


# --- FUNCIÓN DE RE-ANÁLISIS (MODIFICADA PARA DOBLE CONTEXTO) ---

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
    
    llm_result = get_llm_dimensional_scoring(
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
