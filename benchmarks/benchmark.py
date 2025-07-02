import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import onnxruntime as ort
import torch
import pandas as pd

# -------------------------------
#CONFIGURACIÓN INICIAL
# -------------------------------
MODEL_NAME = "distilbert-base-cased-distilled-squad"
ONNX_PATH = "../export/model_qa.onnx"

# -------------------------------
# DDATOS DE PRUEBA
# -------------------------------
CONTEXT = "La Universidad Nacional de Ingeniería está ubicada en Lima, Perú."
QUESTION = "¿Dónde está la UNI?"

#REPETIMOS LAS ENTRADAS PARA SIMULAR UN BATCH
BATCH_SIZE = 16
questions = [QUESTION] * BATCH_SIZE
contexts = [CONTEXT] * BATCH_SIZE

# -------------------------------
#TOKENIZER Y MODELO ORIGINAL (PyTorch)
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pt_model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
pt_model.eval()  #MODO INFERENCIA

# -------------------------------
#MODELO EN ONNX (ONNXRuntime)
# -------------------------------
ort_session = ort.InferenceSession(ONNX_PATH)

# -------------------------------
# FUNCIÓN: TIEMPO DE INFERENCIA PyTorch
# -------------------------------
def benchmark_pytorch():
    inputs = tokenizer(questions, contexts, return_tensors="pt", padding=True, truncation=True)
    start = time.time()
    with torch.no_grad():
        outputs = pt_model(**inputs)
    end = time.time()

    duration = end - start
    latency = duration / BATCH_SIZE
    throughput = BATCH_SIZE / duration

    return {"modelo": "pytorch", "batch_size": BATCH_SIZE, "latencia_s": latency, "throughput": throughput}

# -------------------------------
# Función: tiempo de inferencia ONNX
# -------------------------------
def benchmark_onnx():
    inputs = tokenizer(questions, contexts, return_tensors="np", padding=True, truncation=True)
    start = time.time()
    outputs = ort_session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })
    end = time.time()

    duration = end - start
    latency = duration / BATCH_SIZE
    throughput = BATCH_SIZE / duration

    return {"modelo": "onnx", "batch_size": BATCH_SIZE, "latencia_s": latency, "throughput": throughput}

# -------------------------------
# Ejecución y comparación
# -------------------------------
if __name__ == "__main__":
    results = []
    results.append(benchmark_pytorch())
    results.append(benchmark_onnx())

    # Mostrar resultados como tabla
    df = pd.DataFrame(results)
    print("\nResultados de Benchmark:\n")
    print(df.to_string(index=False))
