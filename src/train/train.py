import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', ''))
from data_loader import load_data

def train_model():
    # Load and preprocess data
    df = load_data('../../data/raw/train.csv')
    # Assuming a 'text' column for simplicity. Replace with your actual text column name.
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']  # Adjust as per your column

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save the model and vectorizer to the outputs directory
    joblib.dump(model, '/app/outputs/models/logistic_regression_model.pkl')
    joblib.dump(vectorizer, '/app/outputs/models/tfidf_vectorizer.pkl')

if __name__ == "__main__":
    train_model()
