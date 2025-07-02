import torch
from transformers import pipeline

# ---------------------------------------------------------------
# CARGAMOS UN PIPELINE DE GENERACIÓN DE TEXTO (MODELO AUTOREGRESIVO)
#ESTE TIPO DE MODELO PREDICE EL SIGUIENTE TOKEN DADO EL ANTERIOR
# ---------------------------------------------------------------
#POR DEFECTO USA 'gpt2', PERO PODEMOS CAMBIARLO A UNO MÁS PEQUEÑO SI ES LENTO
#POR EJEMPLO: 'distilgpt2' ES MÁS LIGERO
text_generator = pipeline("text-generation", model="distilgpt2")

# ---------------------------------------------------------------
# CARGAMOS UN PIPELINE DE PREGUNTAS Y RESPUESTAS (question answering).
#ESTE TIPO DE MODELO SE BASA EN COMPRESIÓN DE TEXTO: RECIBE UN CONTEXTO
#Y UNA PREGUNTA, Y DEVUELVE LA RESPUESTA EXTRAÍDA DEL CONTEXTO
# ---------------------------------------------------------------
# USAMOS 'distilbert-base-cased-distilled-squad' ENTRENADO EN DATASET SQuAD.
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def generate_text(prompt: str, max_length: int = 50) -> str:
    """
    Genera texto a partir de un prompt utilizando un modelo autoregresivo.
    
    Args:
        prompt (str): Texto de entrada que inicia la generación.
        max_length (int): Longitud máxima del texto generado.

    Returns:
        str: Texto generado.
    """
    output = text_generator(prompt, max_length=max_length, num_return_sequences=1)
    return output[0]["generated_text"]

def answer_question(context: str, question: str) -> str:
    """
    Responde una pregunta sobre un contexto dado.

    Args:
        context (str): Texto que contiene la información.
        question (str): Pregunta específica.

    Returns:
        str: Respuesta encontrada en el contexto.
    """
    result = qa_pipeline(question=question, context=context)
    return result["answer"]