from transformers import pipeline

# ---------------------------------------------------------------
# Cargamos un pipeline de generación de texto (modelo autoregresivo).
# Este tipo de modelo predice el siguiente token dado el anterior.
# ---------------------------------------------------------------
# Por defecto usa 'gpt2', pero podemos cambiarlo a uno más pequeño si es lento.
# Por ejemplo: 'distilgpt2' es más ligero.
text_generator = pipeline("text-generation", model="distilgpt2")

# ---------------------------------------------------------------
# Cargamos un pipeline de preguntas y respuestas (question answering).
# Este tipo de modelo se basa en comprensión de texto: recibe un contexto
# y una pregunta, y devuelve la respuesta extraída del contexto.
# ---------------------------------------------------------------
# Usamos 'distilbert-base-cased-distilled-squad' entrenado en dataset SQuAD.
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