from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from typing import List

#CARGAMOS MODELO Y TOKENIZER
MODEL_NAME = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
model.eval()  # MODO INFERENCIA

# ----------------------------------------------------------
# Paso 1: Datos simulados de diferentes longitudes
# ----------------------------------------------------------
contexts = [
    "La UNI está en Lima.",
    "La Universidad Nacional de Ingeniería fue fundada en 1876 y está ubicada en Lima, Perú.",
    "La UNI, también conocida como Universidad Nacional de Ingeniería, es una de las instituciones más importantes del Perú."
]

questions = [
    "¿Dónde está la UNI?",
    "¿Cuándo fue fundada la UNI?",
    "¿Cómo se conoce también a la UNI?"
]

# ----------------------------------------------------------
# Paso 2: Tokenización con padding dinámico
# padding=True ajusta al tamaño máximo del batch automáticamente.
# truncation=True corta si es demasiado largo.
# return_tensors="pt" devuelve tensores para PyTorch.
# ----------------------------------------------------------
inputs = tokenizer(
    questions,
    contexts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# ----------------------------------------------------------
# Paso 3: Inferencia por batch (paralelo)
# El modelo procesa todos los ejemplos a la vez, usando el padding necesario.
# ----------------------------------------------------------
with torch.no_grad():
    outputs = model(**inputs)

# Obtenemos logits de inicio y fin
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# ----------------------------------------------------------
# Paso 4: Decodificamos respuestas token-a-token por ejemplo
# Para cada ejemplo, encontramos el token de inicio y fin de la respuesta.
# ----------------------------------------------------------
for i in range(len(questions)):
    input_ids = inputs["input_ids"][i]
    start_idx = torch.argmax(start_logits[i])
    end_idx = torch.argmax(end_logits[i]) + 1  # El token final es inclusivo

    answer_ids = input_ids[start_idx:end_idx]
    answer = tokenizer.decode(answer_ids)

    print(f"❓ {questions[i]}")
    print(f"✅ Respuesta: {answer}\n")
