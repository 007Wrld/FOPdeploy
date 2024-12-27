from flask import Flask, request, render_template
import requests

app = Flask(__name__)

# Hugging Face API URLs (you need to replace with your own Hugging Face API token)
translation_url = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-XX"  # Use a proper translation model
sentiment_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"  # Sentiment analysis model

# Function to make requests to Hugging Face API
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
            
            # Translate text if not English (Detect language and translate if necessary)
            detected_language = 'en'  # Assume English for simplicity (you can add actual detection here)
            if detected_language != 'en':
                translation_response = query_huggingface_api(input_text, translation_url)
                if "error" in translation_response:
                    return render_template("index.html", sentiment=None, summary=None, original_text=input_text, error="Error during translation")
                translated_text = translation_response[0]['translation_text']
            else:
                translated_text = input_text
            
            # Sentiment analysis
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
