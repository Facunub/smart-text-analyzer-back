from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# 🧠 Cargamos los modelos solo una vez al iniciar la app
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
classifier_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Texto vacío"}), 400

    try:
        # Análisis de sentimiento
        sentiment_result = sentiment_pipeline(text)[0]
        sentiment = sentiment_result["label"]

        # Resumen del texto
        summary_result = summarizer_pipeline(text, max_length=30, min_length=5, do_sample=False)
        summary = summary_result[0]["summary_text"]

        # Clasificación del texto
        categories = ["Consulta", "Reclamo", "Felicitación"]
        classification = classifier_pipeline(text, candidate_labels=categories)
        category = classification["labels"][0]

        # Respuesta sugerida según la categoría
        suggested_replies = {
            "Consulta": "Gracias por tu consulta. En breve te responderemos.",
            "Reclamo": "Lamentamos la experiencia. Revisaremos su caso cuanto antes.",
            "Felicitación": "¡Gracias por tu mensaje positivo!",
        }
        reply = suggested_replies.get(category, "Gracias por contactarnos.")

        return jsonify({
            "sentiment": sentiment,
            "summary": summary,
            "category": category,
            "suggested_reply": reply
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
