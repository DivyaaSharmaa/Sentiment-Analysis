import re
import nltk
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords if not already
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# TextBlob Sentiment
def get_textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Load dataset dynamically
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        if 'text' not in data.columns or 'label' not in data.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")
        data['cleaned_text'] = data['text'].apply(clean_text)
        return data
    except Exception as e:
        print("Error loading dataset:", e)
        return None

# Train ML model dynamically
def train_model(data):
    X = data['cleaned_text']
    y = data['label']
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nML Model trained dynamically!")
    print("Accuracy on test data:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, vectorizer

# Predict sentiment dynamically
def predict_sentiment(text, model, vectorizer):
    cleaned = clean_text(text)
    tb_sentiment = get_textblob_sentiment(cleaned)

    vec = vectorizer.transform([cleaned])
    ml_sentiment = model.predict(vec)[0]

    print(f"\nText: {text}")
    print(f"TextBlob Sentiment: {tb_sentiment}")
    print(f"ML Model Sentiment: {ml_sentiment}")

# Main dynamic interface
if __name__ == "__main__":
    print("==== Dynamic Sentiment Analysis App ====")

    # Load dataset
    dataset_path = input("Enter path to CSV dataset (with 'text' and 'label'): ").strip()
    data = load_dataset(dataset_path)
    if data is None:
        exit()

    # Train ML model
    model, vectorizer = train_model(data)

    # Predict loop
    while True:
        user_input = input("\nEnter a tweet/review (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        predict_sentiment(user_input, model, vectorizer)
