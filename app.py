from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

# Helper function to send requests to Hugging Face Inference API
def query_huggingface_api(model_url, inputs):
    headers = {
        "Authorization": "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"
    }
    response = requests.post(model_url, headers=headers, json={"inputs": inputs})
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Unable to fetch result from Hugging Face API"}

# Helper function for translation using Hugging Face API
def translate_text(text, target_lang="en"):
    model_url = f"https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-{target_lang}-en"
    translation = query_huggingface_api(model_url, text)
    
    if "error" in translation:
        return "Error during translation"
    
    return translation[0]['translation_text']

# Helper function for sentiment analysis using Hugging Face API
def analyze_sentiment(text):
    model_url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"
    sentiment = query_huggingface_api(model_url, text)
    
    if "error" in sentiment:
        return "Error during sentiment analysis"
    
    return sentiment[0]['label']

# Helper function for summarization using Hugging Face API
def summarize_text(text):
    model_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    summary = query_huggingface_api(model_url, text)
    
    if "error" in summary:
        return "Error during summarization"
    
    return summary[0]['summary_text']

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
