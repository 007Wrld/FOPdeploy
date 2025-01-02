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
    'az': 'Azerbaijani',
    'tl': 'Tagalog',
    'sq': 'Albanian',
    'fa': 'Persian',
    'ta': 'Tamil',
    'tr': 'Turkish',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'it': 'Italian',
    'en': 'English',
    'la': 'Latin',
    'ml': 'Malayalam',
    'bs': 'Bosnian',
    'he': 'Hebrew',
    'ta': 'Tamil',
    'pa': 'Punjabi',
    'la': 'Latin',
    'mi': 'Maori',
    'sq': 'Albanian',
    'ga': 'Irish',
    'eo': 'Esperanto',
    'su': 'Sundanese',
    'bn': 'Bengali',
    'no': 'Norwegian',
    'si': 'Sinhala',
    'uk': 'Ukrainian',
    'cy': 'Welsh',
    'km': 'Khmer',
    'ht': 'Haitian Creole',
    'sw': 'Swahili',
    'tl': 'Tagalog',
    'pl': 'Polish',
    'sl': 'Slovenian',
    'tk': 'Turkmen',
    'ro': 'Romanian',
    'el': 'Greek',
    'no': 'Norwegian',
    'la': 'Latin',
    'de': 'German',
    'fr': 'French',
    'en': 'English',
    'tr': 'Turkish',
    'cs': 'Czech',
    'sr': 'Serbian',
    'si': 'Sinhala',
    'sk': 'Slovak',
    'gu': 'Gujarati',
    'pa': 'Punjabi',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'be': 'Belarusian',
    'mr': 'Marathi',
    'az': 'Azerbaijani',
    'am': 'Amharic',
    'ps': 'Pashto',
    'km': 'Khmer',
    'ml': 'Malayalam',
    'te': 'Telugu',
    'or': 'Odia',
    'kn': 'Kannada',
    'zh': 'Chinese',
    'ht': 'Haitian Creole',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'el': 'Greek',
    'hy': 'Armenian',
    'fi': 'Finnish',
    'en': 'English',
    'sv': 'Swedish',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ml': 'Malayalam',
    'fr': 'French',
    'pt': 'Portuguese',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'th': 'Thai',
    'tr': 'Turkish',
    'fi': 'Finnish',
    'sr': 'Serbian',
    'pl': 'Polish',
    'ro': 'Romanian',
    'tr': 'Turkish',
    'en': 'English',
    'ru': 'Russian',
    'he': 'Hebrew',
    'mr': 'Marathi',
    'ja': 'Japanese',
    'fr': 'French',
    'de': 'German',
}


# Helper function to split text into chunks for Google Translate
def split_text_for_translation(text, max_length=5000):
    """
    Splits the text into smaller chunks, each within the Google Translate API limit.
    """
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# Helper function to summarize text using Hugging Face API
def summarize_text_hf(text):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    data = {
        "inputs": text
    }
    try:
        response = requests.post(f"{HF_API_URL}/jaesani/large_eng_summarizer", headers=headers, json=data)
        response.raise_for_status()
        summary = response.json()
        if isinstance(summary, list) and len(summary) > 0:
            return summary[0].get('summary_text', "Error with summarization")
        return "Error with summarization response"
    except requests.exceptions.RequestException as e:
        print(f"Error making API request for summarization: {e}")
        return "Error with summarization API"

# Helper function to call Hugging Face API for sentiment analysis
def analyze_sentiment_hf(text):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    data = {
        "inputs": text
    }
    try:
        response = requests.post(f"{HF_API_URL}/cardiffnlp/twitter-xlm-roberta-base-sentiment", headers=headers, json=data)
        response.raise_for_status()
        sentiment = response.json()
        if isinstance(sentiment, list) and len(sentiment) > 0:
            best_sentiment = max(sentiment[0], key=lambda x: x.get('score', 0))
            return best_sentiment.get('label', 'Unknown'), best_sentiment.get('score', 0)
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
        detected_language_name = LANGUAGES.get(detected_language, detected_language)

        # Translate if not English
        if detected_language != 'en':
            chunks = split_text_for_translation(input_text)
            translated_chunks = []
            for chunk in chunks:
                try:
                    translated_chunks.append(translator.translate(chunk, dest='en').text)
                except Exception as e:
                    print(f"Error translating chunk: {e}")
                    translated_chunks.append(chunk)  # Fallback to original text
            translated_text = " ".join(translated_chunks)
            translated_language = 'English'
        else:
            translated_text = input_text
            translated_language = None

        # Split text into chunks for summarization
        chunks = split_text_into_chunks(translated_text, max_tokens=512)
        summarized_chunks = [summarize_text_hf(chunk) for chunk in chunks]

        # Combine summaries while ensuring total length is under 512 tokens
        combined_summary = " ".join(summarized_chunks)
        if len(combined_summary.split()) > 512:
            combined_summary = " ".join(combined_summary.split()[:512])

        # Get sentiment based on combined summary
        sentiment, sentiment_score = analyze_sentiment_hf(combined_summary)

        # Render results
        return render_template(
            "index.html",
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            summary=combined_summary,
            original_text=input_text,
            translated_text=translated_text,
            detected_language_name=detected_language_name,
            translated_language=translated_language
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
