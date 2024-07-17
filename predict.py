import joblib
import sys

def load_model():
    model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('spam_detector_vectorizer.pkl')
    return model, vectorizer


def predict(text):
    model, vectorizer = load_model()
    text_transformed = vectorizer.transform([text])
    prediciton = model.predict(text_transformed)
    return "Spam" if prediciton[0] == 1 else "Not Spam"

if __name__ == "__main__":
    text = sys.argv[1]
    prediction = predict(text)
    print(prediction)