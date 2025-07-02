from fastapi import FastAPI
from pydantic import BaseModel
from time import time
from inference.pipelines import generate_text, answer_question

# -------------------------------
# DEFINIMOS LA APP FastAPI
# -------------------------------
app = FastAPI(
    title="API de Inferencia NLP",
    description="Permite generar texto y responder preguntas usando modelos Hugging Face y ONNX.",
    version="1.0.0"
)

# -------------------------------
#ESQUEMAS DE ENTRADA (inputs)
# -------------------------------
class PromptInput(BaseModel):
    prompt: str
    max_length: int = 50

class QAInput(BaseModel):
    context: str
    question: str

# -------------------------------
# ENDPOINT: /generate (POST)
# -------------------------------
@app.post("/generate")
def generate_endpoint(data: PromptInput):
    start = time()
    result = generate_text(prompt=data.prompt, max_length=data.max_length)
    duration = round((time() - start) * 1000, 2)  # en milisegundos
    return {
        "input": data.prompt,
        "output": result,
        "latencia_ms": duration
    }

# -------------------------------
# ENDPOINT: /qa (POST)
# -------------------------------
@app.post("/qa")
def qa_endpoint(data: QAInput):
    start = time()
    answer = answer_question(context=data.context, question=data.question)
    duration = round((time() - start) * 1000, 2)
    return {
        "pregunta": data.question,
        "respuesta": answer,
        "latencia_ms": duration
    }

# -------------------------------
# ROOT: MENSAJE DE BIENVENIDAA
# -------------------------------
@app.get("/")
def root():
    return {"mensaje": "Bienvenido a la API NLP con FastAPI. Visita /docs para probarla."}
