from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from googletrans import Translator
from deep_translator import GoogleTranslator
import os

app = Flask(__name__)

# Declare global variables for models and tokenizers but don't load them yet
tokenizer_sentiment = None
model_sentiment = None

tokenizer_summarizer = None
model_summarizer = None

translator = Translator()

# Lazy load sentiment model
def load_sentiment_model():
    global tokenizer_sentiment, model_sentiment
    if tokenizer_sentiment is None or model_sentiment is None:
        print("Loading sentiment analysis model...")
        tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
        model_sentiment = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    return tokenizer_sentiment, model_sentiment

# Lazy load summarization model
def load_summarizer_model():
    global tokenizer_summarizer, model_summarizer
    if tokenizer_summarizer is None or model_summarizer is None:
        print("Loading summarization model...")
        tokenizer_summarizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model_summarizer = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    return tokenizer_summarizer, model_summarizer

# Fallback translation function
def safe_translate(text, target="en"):
    try:
        return translator.translate(text, dest=target).text
    except Exception as e:
        print(f"Googletrans failed, using Deep Translator: {e}")
        try:
            return GoogleTranslator(source="auto", target=target).translate(text)
        except Exception as fallback_error:
            raise RuntimeError(f"Translation failed: {fallback_error}")

# Helper function to summarize text
def summarize_text(text):
    # Lazy load summarization model
    tokenizer_summarizer, model_summarizer = load_summarizer_model()
    
    inputs = tokenizer_summarizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary = model_summarizer.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    summary_text = tokenizer_summarizer.decode(summary[0], skip_special_tokens=True)
    return summary_text

# Helper function for sentiment analysis
def analyze_sentiment(text):
    # Lazy load sentiment model
    tokenizer_sentiment, model_sentiment = load_sentiment_model()
    
    inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model_sentiment(**inputs)
    logits = outputs.logits
    sentiment = logits.argmax().item()
    sentiment_label = ["negative", "neutral", "positive"][sentiment]
    return sentiment_label

# Route to handle the form and display results
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_text = request.form.get("text", "")

        # Handle empty input
        if not input_text.strip():
            return render_template("index.html", error="Please enter some text.", sentiment=None, summary=None, original_text=None, translated_text=None)

        try:
            # Detect the language of the text
            detected_language = translator.detect(input_text).lang

            # If the language is not English, translate it
            if detected_language != 'en':
                translated_text = safe_translate(input_text, target='en')
            else:
                translated_text = input_text

            # Summarize the text (only if it's longer than 250 words)
            if len(translated_text.split()) > 250:
                summarized_text = summarize_text(translated_text)
            else:
                summarized_text = translated_text

            # Get sentiment (based on summarized or original text)
            sentiment = analyze_sentiment(summarized_text)

            return render_template(
                "index.html", sentiment=sentiment, summary=summarized_text, 
                original_text=input_text, translated_text=translated_text, error=None
            )
        except Exception as e:
            return render_template("index.html", error=f"An error occurred: {e}", sentiment=None, summary=None, original_text=None, translated_text=None)

    return render_template("index.html", sentiment=None, summary=None, original_text=None, translated_text=None, error=None)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
