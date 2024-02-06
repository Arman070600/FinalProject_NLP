import joblib
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', ''))
from data_loader import load_data

def predict_sentiment(text):
    model = joblib.load('/app/outputs/models/logistic_regression_model.pkl')
    vectorizer = joblib.load('/app/outputs/models/tfidf_vectorizer.pkl')

    transformed_text = vectorizer.transform([text])

    prediction = model.predict(transformed_text)
    return prediction

if __name__ == "__main__":
    input_text = "This is a great movie!"
    prediction = predict_sentiment(input_text)
    print(f"Predicted Sentiment: {prediction}")
