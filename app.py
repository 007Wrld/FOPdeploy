from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from googletrans import Translator
import requests

app = Flask(__name__)

# Define your Hugging Face API token and URL
HF_API_TOKEN = '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'
HF_API_URL = 'https://api-inference.huggingface.co/models'

# Load models
tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("jaesani/large_eng_summarizer")
summarizer_tokenizer = AutoTokenizer.from_pretrained("jaesani/large_eng_summarizer")
translator = Translator()

# Helper function to summarize text
def summarize_text(text):
    inputs = summarizer_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary = summarizer_model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    summary_text = summarizer_tokenizer.decode(summary[0], skip_special_tokens=True)
    return summary_text

# Helper function to call Hugging Face API for sentiment analysis
def analyze_sentiment_hf(text):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    data = {
        "inputs": text
    }
    try:
        # Make the API request to the sentiment analysis model
        response = requests.post(f"{HF_API_URL}/cardiffnlp/twitter-xlm-roberta-base-sentiment", headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        sentiment = response.json()
        
        # Print the raw response to inspect its structure
        print("Sentiment response:", sentiment)

        # Access the first list and then the first dictionary
        if isinstance(sentiment, list) and len(sentiment) > 0:
            sentiment_result = sentiment[0]  # Get the first list in the response
            
            # Print the structure of the first item
            print("Sentiment result structure:", sentiment_result)
            
            if isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                # Extract the label of the first sentiment in the list
                return sentiment_result[0].get('label', 'No label found')
            else:
                print("Unexpected structure in sentiment result:", sentiment_result)
                return "Error with sentiment label"
        else:
            print("Unexpected response format:", sentiment)
            return "Error with sentiment analysis response"
    
    except requests.exceptions.RequestException as e:
        print(f"Error making API request for sentiment analysis: {e}")
        return "Error with sentiment analysis API"

# Route to handle the form and display results
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_text = request.form["text"]
        
        # Detect the language of the text
        detected_language = translator.detect(input_text).lang
        
        # If the language is not English, translate it
        if detected_language != 'en':
            translated_text = translator.translate(input_text, dest='en').text
        else:
            translated_text = input_text
        
        # Summarize the text (only if it's longer than 250 words)
        if len(translated_text.split()) > 250:
            summarized_text = summarize_text(translated_text)
        else:
            summarized_text = translated_text
        
        # Get sentiment (based on summarized or original text)
        sentiment = analyze_sentiment_hf(summarized_text)
        
        return render_template("index.html", sentiment=sentiment, summary=summarized_text, original_text=input_text)
    
    return render_template("index.html", sentiment=None, summary=None, original_text=None)

if __name__ == "__main__":
    app.run(debug=True)
