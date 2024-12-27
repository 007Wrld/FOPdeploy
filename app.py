from flask import Flask, request, jsonify, render_template
import requests
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Initialize the Hugging Face translation model
translation_model_name = "Helsinki-NLP/opus-mt-en-x"  # English to many languages
tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
model = MarianMTModel.from_pretrained(translation_model_name)

# Helper function for translation using Hugging Face model
def translate_text(text, target_lang="en"):
    # Translate text to the target language using Hugging Face's MarianMT
    # 'en' represents English, but this model supports multiple languages
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True, truncation=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Helper function for sentiment analysis using Hugging Face API
def analyze_sentiment(text):
    sentiment_model_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"
    headers = {"Authorization": "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"}
    
    response = requests.post(sentiment_model_url, headers=headers, json={"inputs": text})
    
    if response.status_code == 200:
        sentiment = response.json()[0]['label']
        return sentiment
    else:
        return "Error: Unable to fetch sentiment"

# Helper function for summarization using Hugging Face API
def summarize_text(text):
    summarization_model_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": "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"}
    
    response = requests.post(summarization_model_url, headers=headers, json={"inputs": text})
    
    if response.status_code == 200:
        summary = response.json()[0]['summary_text']
        return summary
    else:
        return "Error: Unable to fetch summary"

@app.route("/", methods=["GET", "POST"])
def home():
    try:
        if request.method == "POST":
            input_text = request.form["text"]
            
            # Detect language and translate to English if needed
            # (For simplicity, let's assume we're always translating to English)
            translated_text = translate_text(input_text, target_lang="en")
            
            # Summarize text (if it's long enough)
            if len(translated_text.split()) > 250:
                summarized_text = summarize_text(translated_text)
            else:
                summarized_text = translated_text
            
            # Analyze sentiment
            sentiment = analyze_sentiment(summarized_text)
            
            return render_template("index.html", sentiment=sentiment, summary=summarized_text, original_text=input_text)
        
        return render_template("index.html", sentiment=None, summary=None, original_text=None)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
