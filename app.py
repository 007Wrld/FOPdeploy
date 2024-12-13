import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from googletrans import Translator

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

# Helper function for sentiment analysis
def analyze_sentiment(text):
    inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model_sentiment(**inputs)
    logits = outputs.logits
    sentiment = logits.argmax().item()
    sentiment_label = ["negative", "neutral", "positive"][sentiment]
    return sentiment_label

# Streamlit app layout
st.title("FOP Sentiment Analysis")
input_text = st.text_area("Enter your text here...", height=150)

if input_text:
    # Detect the language of the text
    detected_language = translator.detect(input_text).lang

    # Translate text if not English
    if detected_language != 'en':
        translated_text = translator.translate(input_text, dest='en').text
        st.write("Translated Text:")
        st.write(translated_text)
    else:
        translated_text = input_text

    # Summarize the text if it's longer than 250 words
    if len(translated_text.split()) > 250:
        summarized_text = summarize_text(translated_text)
        st.write("Summary:")
        st.write(summarized_text)
    else:
        summarized_text = translated_text

    # Get sentiment of the summarized or original text
    sentiment = analyze_sentiment(summarized_text)
    st.write(f"Sentiment: {sentiment.capitalize()}")
