from flask import Flask, request, render_template
from googletrans import Translator  # Use googletrans for translation
import requests

app = Flask(__name__)

# Initialize translator (googletrans)
translator = Translator()

# Hugging Face Sentiment Analysis model URL
sentiment_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"  # Sentiment analysis model

# Function to make requests to Hugging Face API for sentiment analysis
def query_huggingface_api(input_text, model_url):
    headers = {
        "Authorization": "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"  # Replace with your Hugging Face API key
    }
    payload = {"inputs": input_text}
    response = requests.post(model_url, headers=headers, json=payload)
    return response.json()

@app.route("/", methods=["GET", "POST"])
def home():
    try:
        if request.method == "POST":
            input_text = request.form["text"]
            
            # Detect the language of the input text using googletrans
            detected_language = translator.detect(input_text).lang
            
            # Translate if the text is not in English
            if detected_language != 'en':
                translated_text = translator.translate(input_text, dest='en').text
            else:
                translated_text = input_text
            
            # Sentiment analysis using Hugging Face
            sentiment_response = query_huggingface_api(translated_text, sentiment_url)
            if "error" in sentiment_response:
                return render_template("index.html", sentiment=None, summary=None, original_text=input_text, error="Error during sentiment analysis")
            
            sentiment = sentiment_response[0]['label']
            return render_template("index.html", sentiment=sentiment, summary=translated_text, original_text=input_text)
        
        return render_template("index.html", sentiment=None, summary=None, original_text=None)
    except Exception as e:
        return render_template("index.html", sentiment=None, summary=None, original_text=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
