import requests
from flask import Flask, request, jsonify, render_template
from googletrans import Translator

app = Flask(__name__)

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models"
HF_API_TOKEN = "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"  # Replace with your token

# Helper function to call Hugging Face API for sentiment analysis
def analyze_sentiment_hf(text):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    data = {
        "inputs": text
    }
    # Send a request to Hugging Face API for sentiment analysis
    response = requests.post(f"{HF_API_URL}/cardiffnlp/twitter-xlm-roberta-base-sentiment", headers=headers, json=data)
    sentiment = response.json()
    return sentiment[0]['label']

# Helper function to call Hugging Face API for text summarization
def summarize_text_hf(text):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    data = {
        "inputs": text
    }
    # Send a request to Hugging Face API for summarization
    response = requests.post(f"{HF_API_URL}/jaesani/large_eng_summarizer", headers=headers, json=data)
    summary = response.json()
    return summary[0]['summary_text']

# Route to handle the form and display results
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_text = request.form["text"]
        
        # Detect the language of the text
        translator = Translator()
        detected_language = translator.detect(input_text).lang
        
        # If the language is not English, translate it
        if detected_language != 'en':
            translated_text = translator.translate(input_text, dest='en').text
        else:
            translated_text = input_text
        
        # Summarize the text (only if it's longer than 250 words)
        if len(translated_text.split()) > 250:
            summarized_text = summarize_text_hf(translated_text)
        else:
            summarized_text = translated_text
        
        # Get sentiment (based on summarized or original text)
        sentiment = analyze_sentiment_hf(summarized_text)
        
        return render_template("index.html", sentiment=sentiment, summary=summarized_text, original_text=input_text)
    
    return render_template("index.html", sentiment=None, summary=None, original_text=None)

if __name__ == "__main__":
    app.run(debug=True)
