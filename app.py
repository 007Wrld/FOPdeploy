from flask import Flask, request, jsonify, render_template
from googletrans import Translator
import requests

app = Flask(__name__)

# Define your Hugging Face API token and URL
HF_API_TOKEN = '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'
HF_API_URL = 'https://api-inference.huggingface.co/models'

translator = Translator()

# Language code to full name mapping
LANGUAGES = {
    'af': 'Afrikaans',
    'sq': 'Albanian',
    'ar': 'Arabic',
    'hy': 'Armenian',
    'bn': 'Bengali',
    'bs': 'Bosnian',
    'ca': 'Catalan',
    'hr': 'Croatian',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English',
    'eo': 'Esperanto',
    'et': 'Estonian',
    'tl': 'Filipino (Tagalog)',
    'ceb': 'Cebuano',
    'fi': 'Finnish',
    'fr': 'French',
    'de': 'German',
    'el': 'Greek',
    'gu': 'Gujarati',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'is': 'Icelandic',
    'id': 'Indonesian',
    'it': 'Italian',
    'ja': 'Japanese',
    'jw': 'Javanese',
    'ka': 'Georgian',
    'km': 'Khmer',
    'ko': 'Korean',
    'la': 'Latin',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'mk': 'Macedonian',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'ne': 'Nepali',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sr': 'Serbian',
    'si': 'Sinhala',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'es': 'Spanish',
    'su': 'Sundanese',
    'sw': 'Swahili',
    'sv': 'Swedish',
    'ta': 'Tamil',
    'te': 'Telugu',
    'th': 'Thai',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'cy': 'Welsh',
    'xh': 'Xhosa',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'he': 'Hebrew',
    'pa': 'Punjabi',
    'sd': 'Sindhi',
    'ilo': 'Ilokano',
    'az': 'Azerbaijani'
}

# Function to split text into chunks

def split_text(text, max_tokens=512):
    """
    Splits text into chunks that fit within the model's token limit.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Helper function to summarize text using Hugging Face API
def summarize_text_hf(text):
    """
    Summarizes text by breaking it into smaller chunks and summarizing each chunk.
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    summaries = []
    try:
        chunks = split_text(text)
        for chunk in chunks:
            data = {"inputs": chunk}
            response = requests.post(f"{HF_API_URL}/jaesani/large_eng_summarizer", headers=headers, json=data)
            response.raise_for_status()
            summary = response.json()
            if isinstance(summary, list) and len(summary) > 0:
                summaries.append(summary[0].get('summary_text', "Error with summarization"))
        return " ".join(summaries)
    except requests.exceptions.RequestException as e:
        print(f"Error making API request for summarization: {e}")
        return "Error with summarization API"

# Helper function to call Hugging Face API for sentiment analysis
def analyze_sentiment_hf(text):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    data = {"inputs": text}
    try:
        response = requests.post(f"{HF_API_URL}/cardiffnlp/twitter-xlm-roberta-base-sentiment", headers=headers, json=data)
        response.raise_for_status()
        sentiment = response.json()
        if isinstance(sentiment, list) and len(sentiment) > 0:
            sentiment_list = sentiment[0]
            if isinstance(sentiment_list, list):
                best_sentiment = max(sentiment_list, key=lambda x: x.get('score', 0))
                sentiment_label = best_sentiment.get('label', 'Unknown sentiment')
                sentiment_score = best_sentiment.get('score', 0)
                return sentiment_label, sentiment_score
        return "Unexpected response format", 0
    except requests.exceptions.RequestException as e:
        print(f"Error making API request for sentiment analysis: {e}")
        return "Error with sentiment analysis API", 0

# Route to handle the form and display results
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_text = request.form["text"]

        # Detect the language of the text
        detected_language = translator.detect(input_text).lang

        # Convert the detected language code to full name
        detected_language_name = LANGUAGES.get(detected_language, detected_language)

        # If the language is not English, translate it
        if detected_language != 'en':
            translated_text = translator.translate(input_text, dest='en').text
            translated_language = 'English'  # Since we always translate to English
        else:
            translated_text = input_text
            translated_language = None  # No translation if the text is already in English

        # Summarize if the translated text exceeds 500 tokens
        token_limit = 500
        if len(translated_text.split()) > token_limit:
            summarized_text = summarize_text_hf(translated_text)
        else:
            summarized_text = translated_text

        # Perform sentiment analysis
        sentiment, sentiment_score = analyze_sentiment_hf(summarized_text)

        # Return the results
        return render_template(
            "index.html",
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            summary=summarized_text,
            original_text=input_text,
            translated_text=translated_text,
            detected_language_name=detected_language_name,
            translated_language=translated_language
        )

    return render_template(
        "index.html",
        sentiment=None,
        sentiment_score=None,
        summary=None,
        original_text=None,
        translated_text=None,
        detected_language_name=None,
        translated_language=None
    )

if __name__ == "__main__":
    app.run(debug=True)
