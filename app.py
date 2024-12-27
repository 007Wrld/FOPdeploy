from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from googletrans import Translator

app = Flask(__name__)

# Initialize variables for models and tokenizer
tokenizer_sentiment = None
model_sentiment = None
summarizer_model = None
summarizer_tokenizer = None
translator = Translator()

# Lazy load models
def load_sentiment_model():
    global tokenizer_sentiment, model_sentiment
    if tokenizer_sentiment is None or model_sentiment is None:
        tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
        model_sentiment = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

def load_summarizer_model():
    global summarizer_model, summarizer_tokenizer
    if summarizer_model is None or summarizer_tokenizer is None:
        summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("jaesani/large_eng_summarizer")
        summarizer_tokenizer = AutoTokenizer.from_pretrained("jaesani/large_eng_summarizer")

# Helper function to summarize text
def summarize_text(text):
    inputs = summarizer_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary = summarizer_model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    return summarizer_tokenizer.decode(summary[0], skip_special_tokens=True)

# Helper function for sentiment analysis
def analyze_sentiment(text):
    inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model_sentiment(**inputs)
    logits = outputs.logits
    sentiment = logits.argmax().item()
    return ["negative", "neutral", "positive"][sentiment]

@app.route("/", methods=["GET", "POST"])
def home():
    try:
        if request.method == "POST":
            input_text = request.form["text"]
            
            # Detect the language of the text
            detected_language = translator.detect(input_text).lang
            
            # Translate if needed
            translated_text = translator.translate(input_text, dest='en').text if detected_language != 'en' else input_text
            
            # Summarize text
            if len(translated_text.split()) > 250:
                load_summarizer_model()
                summarized_text = summarize_text(translated_text)
            else:
                summarized_text = translated_text
            
            # Analyze sentiment
            load_sentiment_model()
            sentiment = analyze_sentiment(summarized_text)
            
            return render_template("index.html", sentiment=sentiment, summary=summarized_text, original_text=input_text)
        
        return render_template("index.html", sentiment=None, summary=None, original_text=None)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
