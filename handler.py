import os
import requests
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from googletrans import Translator

app = Flask(__name__)

# Hugging Face API token and URL from environment variables
HF_API_TOKEN = os.getenv('HF_API_TOKEN', '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')  # Default for local testing
HF_API_URL = 'https://api-inference.huggingface.co/models'

# Load models
tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("jaesani/large_eng_summarizer")
summarizer_tokenizer = AutoTokenizer.from_pretrained("jaesani/large_eng_summarizer")
translator = Translator()

def summarize_text(text):
    inputs = summarizer_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary = summarizer_model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    summary_text = summarizer_tokenizer.decode(summary[0], skip_special_tokens=True)
    return summary_text

def analyze_sentiment_hf(text):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    data = {"inputs": text}
    try:
        response = requests.post(f"{HF_API_URL}/cardiffnlp/twitter-xlm-roberta-base-sentiment", headers=headers, json=data)
        response.raise_for_status()  # Check for errors
        sentiment = response.json()
        
        if isinstance(sentiment, list) and len(sentiment) > 0:
            sentiment_result = sentiment[0]  # First sentiment result
            if isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                return sentiment_result[0].get('label', 'No label found')
            else:
                return "Error with sentiment label"
        else:
            return "Unexpected response format"
    except requests.exceptions.RequestException as e:
        return f"Error with sentiment analysis: {e}"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_text = request.form["text"]
        detected_language = translator.detect(input_text).lang
        
        if detected_language != 'en':
            translated_text = translator.translate(input_text, dest='en').text
        else:
            translated_text = input_text
        
        if len(translated_text.split()) > 250:
            summarized_text = summarize_text(translated_text)
        else:
            summarized_text = translated_text
        
        sentiment = analyze_sentiment_hf(summarized_text)
        
        return render_template("index.html", sentiment=sentiment, summary=summarized_text, original_text=input_text)
    
    return render_template("index.html", sentiment=None, summary=None, original_text=None)

# Make sure Vercel knows how to handle the Flask app
def handler(request):
    with app.app_context():
        return app.full_dispatch_request()

if __name__ == "__main__":
    app.run(debug=True)
