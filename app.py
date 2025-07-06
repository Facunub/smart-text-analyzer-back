import gradio as gr
from transformers import pipeline

# Cargamos los modelos UNA VEZ
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
classifier_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def analyze_text(text):
    if not text.strip():
        return "N/A", "N/A", "N/A", "Texto vacío."

    # Sentimiento
    sentiment = sentiment_pipeline(text)[0]["label"]

    # Resumen
    summary = summarizer_pipeline(text, max_length=30, min_length=5, do_sample=False)[0]["summary_text"]

    # Clasificación
    labels = ["Consulta", "Reclamo", "Felicitación"]
    result = classifier_pipeline(text, candidate_labels=labels)
    category = result["labels"][0]

    # Respuesta sugerida
    replies = {
        "Consulta": "Gracias por tu consulta. En breve te responderemos.",
        "Reclamo": "Lamentamos la experiencia. Revisaremos su caso cuanto antes.",
        "Felicitación": "¡Gracias por tu mensaje positivo!",
    }
    suggested_reply = replies.get(category, "Gracias por contactarnos.")

    return sentiment, summary, category, suggested_reply

# Interfaz Gradio
interface = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(lines=5, placeholder="Escribe aquí el texto a analizar..."),
    outputs=[
        gr.Textbox(label="Sentimiento"),
        gr.Textbox(label="Resumen"),
        gr.Textbox(label="Categoría"),
        gr.Textbox(label="Respuesta Sugerida"),
    ],
    title="Análisis Inteligente de Texto",
    description="Detecta sentimiento, categoría, resumen y respuesta sugerida."
)

interface.launch()
