import requests
from flask import Flask, request, jsonify, render_template
from googletrans import Translator

app = Flask(__name__)

# Define your Hugging Face API token and URL
HF_API_TOKEN = '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'
HF_API_URL = 'https://api-inference.huggingface.co/models'

translator = Translator()

# Function to chunk text into smaller parts based on token count (500 tokens)
def chunk_text(text, max_tokens=500):
    words = text.split()  # Split the text into words
    chunks = []
    chunk = []
    token_count = 0

    for word in words:
        token_count += len(word.split())  # Count tokens (words in this case)
        if token_count > max_tokens:
            chunks.append(" ".join(chunk))  # Join chunk into a string and add to list
            chunk = [word]  # Start a new chunk
            token_count = len(word.split())  # Reset token count for new chunk
        else:
            chunk.append(word)
    
    if chunk:
        chunks.append(" ".join(chunk))  # Append the final chunk
    
    return chunks

# Function to get sentiment from Hugging Face
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
            sentiment_list = sentiment[0]
            if isinstance(sentiment_list, list):
                best_sentiment = max(sentiment_list, key=lambda x: x.get('score', 0))
                sentiment_label = best_sentiment.get('label', 'Unknown sentiment')
                sentiment_score = best_sentiment.get('score', 0)
                return sentiment_label, sentiment_score
        return "Unexpected response format", 0
    except requests.exceptions.RequestException as e:
        print(f"Error with sentiment analysis: {e}")
        return "Error", 0

# Route for processing the input text and displaying results
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_text = request.form["text"]

        # Detect the language of the input text
        detected_language = translator.detect(input_text).lang
        detected_language_name = LANGUAGES.get(detected_language, detected_language)

        # Translate to English if the language is not English
        if detected_language != 'en':
            translated_text = translator.translate(input_text, dest='en').text
            translated_language = 'English'
        else:
            translated_text = input_text
            translated_language = None

        # Chunk text if it's long
        chunks = chunk_text(translated_text)

        # Analyze sentiment for each chunk and aggregate the results
        sentiment_labels = []
        sentiment_scores = []

        for chunk in chunks:
            sentiment, sentiment_score = analyze_sentiment_hf(chunk)
            sentiment_labels.append(sentiment)
            sentiment_scores.append(sentiment_score)

        # Aggregate sentiment (for simplicity, using the most frequent sentiment label)
        sentiment = max(set(sentiment_labels), key=sentiment_labels.count)
        sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

        # Return the result to the user
        return render_template("index.html", sentiment=sentiment, sentiment_score=sentiment_score, 
                               original_text=input_text, translated_text=translated_text, 
                               detected_language_name=detected_language_name, translated_language=translated_language)

    return render_template("index.html", sentiment=None, sentiment_score=None)

if __name__ == "__main__":
    app.run(debug=True)
