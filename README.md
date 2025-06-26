# Proyecto 4: Pipeline de Inferencia y Optimización 

## Contexto y Motivación

Más allá del entrenamiento de modelos de lenguaje, su despliegue en producción requiere optimizar latencia (tiempo de respuesta) y throughput (número de peticiones por segundo). Este proyecto explora:

- El uso de `pipelines` de Hugging Face (`transformers`)
- La exportación de modelos a formato ONNX (para mejorar el rendimiento en CPU)
- La integración en un microservicio con FastAPI


## Objetivos

1. **Definir pipelines de inferencia**:
   - Generación de texto (`pipeline("text-generation")`)
   - Preguntas y respuestas (`pipeline("question-answering")`)

2. **Exportar a ONNX**:
   - Convertir modelos ligeros (ej. `DistilBERT`) a formato ONNX y medir tiempo de inferencia.

3. **Implementar esquemas de batching**:
   - Autoregressive token-a-token.
   - Batch completo con padding dinámico.

4. **Desarrollar un servidor REST con FastAPI**:
   - Rutas: `/generate` y `/qa`
   - Medir latencia y throughput en Swagger, `curl` o Postman

Requirements:

pip install -r requirements.txt


*Jairo Andre*