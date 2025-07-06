from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Cargar modelos
sentiment_model = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def classify_category(text):
    text = text.lower()
    if "precio" in text or "factura" in text:
        return "Facturación"
    elif "reclamo" in text or "molesto" in text:
        return "Reclamo"
    elif "ayuda" in text or "consulta" in text:
        return "Consulta"
    else:
        return "General"

def generate_reply(category):
    if category == "Reclamo":
        return "Lamentamos la experiencia. Revisaremos su caso cuanto antes."
    elif category == "Facturación":
        return "Nuestro equipo de facturación revisará su solicitud."
    elif category == "Consulta":
        return "Gracias por su consulta. Le responderemos pronto."
    else:
        return "Gracias por contactarnos. Estamos trabajando en su requerimiento."

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")

    sentiment = sentiment_model(text)[0]["label"]
    summary = summarizer(text, max_length=40, min_length=10, do_sample=False)[0]["summary_text"]
    category = classify_category(text)
    reply = generate_reply(category)

    return jsonify({
        "sentiment": sentiment,
        "summary": summary,
        "category": category,
        "suggested_reply": reply
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

