import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from config import config

def load_and_preprocess_data(filename=config.DATA_FILE):
    df = pd.read_csv(filename)
    df['processed_review'] = df['review'].apply(preprocess_text)
    
    sentiment_map = {'negative': 0, 'positive': 1}
    texts = df['processed_review'].values
    labels = df['sentiment'].map(sentiment_map).values
    
    return texts, labels

def preprocess_text(text):
    text = strip_html(text)
    text = remove_special_characters(text)
    text = text.lower()
    text = stem_text(text)
    text = remove_stopwords(text)
    # Limit the sequence length
    words = text.split()[:config.MAX_SEQUENCE_LENGTH]
    return ' '.join(words)

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_special_characters(text, remove_digits=config.REMOVE_DIGITS):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def stem_text(text):
    if config.STEM_WORDS:
        ps = PorterStemmer()
        return ' '.join([ps.stem(word) for word in text.split()])
    return text

def remove_stopwords(text):
    if config.REMOVE_STOPWORDS:
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in text.split() if word not in stop_words])
    return text

def build_vocabulary(texts):
    vectorizer = CountVectorizer(max_features=config.MAX_VOCAB_SIZE, ngram_range=config.NGRAM_RANGE)
    vectorizer.fit(texts)
    return vectorizer.vocabulary_, vectorizer

def vectorize_data(texts, vectorizer):
    if config.USE_TFIDF:
        tfidf_vectorizer = TfidfVectorizer(vocabulary=vectorizer.vocabulary_, ngram_range=config.NGRAM_RANGE)
        return tfidf_vectorizer.fit_transform(texts)
    else:
        return vectorizer.transform(texts)

def process_in_chunks(X, y, chunk_size=10000):
    for i in range(0, len(X), chunk_size):
        yield X[i:i+chunk_size], y[i:i+chunk_size]
