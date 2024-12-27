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
    try:
        response = requests.post(model_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()  # Return the response JSON if successful
    except requests.exceptions.RequestException as e:
        return {"error": f"Sentiment API request failed: {e}"}

@app.route("/", methods=["GET", "POST"])
def home():
    try:
        if request.method == "POST":
            input_text = request.form["text"]
            
            # Detect the language of the input text using googletrans
            detected_language = translator.detect(input_text).lang
            print(f"Detected Language: {detected_language}")  # Debugging output

            # Translate if the text is not in English
            try:
                if detected_language != 'en':
                    translated_text = translator.translate(input_text, dest='en').text
                else:
                    translated_text = input_text
                print(f"Translated Text: {translated_text}")  # Debugging output
            except Exception as e:
                translated_text = None
                print(f"Translation Error: {e}")  # Debugging output
                return render_template("index.html", sentiment=None, summary=None, original_text=input_text, error="Error during translation")

            # Sentiment analysis using Hugging Face
            sentiment_response = query_huggingface_api(translated_text, sentiment_url)
            if "error" in sentiment_response:
                print(f"Sentiment Analysis Error: {sentiment_response['error']}")  # Debugging output
                return render_template("index.html", sentiment=None, summary=None, original_text=input_text, error=sentiment_response['error'])
            
            if not sentiment_response:  # Check if the response is empty or invalid
                return render_template("index.html", sentiment=None, summary=None, original_text=input_text, error="Received empty response from sentiment API")

            sentiment = sentiment_response[0].get('label', 'Unknown')  # Safely extract sentiment label
            return render_template("index.html", sentiment=sentiment, summary=translated_text, original_text=input_text)
        
        return render_template("index.html", sentiment=None, summary=None, original_text=None)
    except Exception as e:
        print(f"General Error: {e}")  # Debugging output
        return render_template("index.html", sentiment=None, summary=None, original_text=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
