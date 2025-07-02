from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.onnx import export
from transformers.onnx import FeaturesManager
from pathlib import Path
import onnxruntime as ort
import torch

# ----------------------------------------------------------
# PASO 1: CONFIGURACIONES BÁSICAS
# ----------------------------------------------------------
MODEL_NAME = "distilbert-base-cased-distilled-squad"  # MODELO PARA QA
ONNX_PATH = Path("export/model_qa.onnx")              # RUTA DONDE SE GUARDA EL MODELO ONNX

# ----------------------------------------------------------
# PASO 2: CARGAMOS EL TOKENIZER Y MODELO EN PyTorch
# ----------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# ----------------------------------------------------------
# PASO 3: PREPARAMOS EL MODELO PARA EXPORTARLO
# SE USA EL GESTOR DE CARACTERÍSTICAS PARA SABER QUÉ SECCIONES
# EXPORTAR DEL MODELO SEGÚN SU TIPO (EN ESTE CASO QA).
# ----------------------------------------------------------
feature = "question-answering"
framework = "pt"  #EXPORTAMOS DESDE PyTorch

# OBTENEMOS EL CONFIG PARA LA EXPORTACIÓN
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = model_onnx_config(model.config)

#CREAMOS UN EJEMPLO DE ENTRADA (dummy input)
dummy_inputs = onnx_config.generate_dummy_inputs(tokenizer, framework=framework)

# ----------------------------------------------------------
# PASO 4: EXPORTAMOS A ONNX
# ----------------------------------------------------------
export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_config,
    opset=17,
    output=ONNX_PATH,
    #tokenizer=tokenizer,
    #inputs=dummy_inputs,
    #framework=framework
    device="cpu"
)

print(f"Modelo exportado a ONNX en {ONNX_PATH.resolve()}")

# ----------------------------------------------------------
# PASO 5: CARGA Y PRUEBA CON ONNXRuntime
# ----------------------------------------------------------
#CREAMOS UN INPUT REAL DE PRUEBA
context = "La Universidad Nacional de Ingeniería está en Lima, Perú."
question = "¿Dónde está la UNI?"
inputs = tokenizer(question, context, return_tensors="np")

# CREAMOS UNA SESIÓN DE ONNXRuntime
session = ort.InferenceSession(str(ONNX_PATH))

# EJECUTAMOS INFERENCIA
outputs = session.run(None, {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"]
})

start_logits, end_logits = outputs
start = start_logits.argmax()
end = end_logits.argmax() + 1

#DECODIFICAMOS LA RESPUESTA
answer = tokenizer.decode(inputs["input_ids"][0][start:end])
print(f"Respuesta con ONNX: {answer}")