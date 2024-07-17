import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.naive_bayes import MultinomialNB


def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    data = pd.read_csv(url, sep='\t', header=None, names=['label','text' ])
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data


def train_model(data):
    x = data['text']
    y = data['label']
    vectorizer = TfidfVectorizer()#understand why I used this model
    x = vectorizer.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2) #CHECK OUT WHAT THIS IS
    model = MultinomialNB() #CHECK OUT WHAT THIS IS
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    joblib.dump(model, 'spam_detector_model.pkl')
    joblib.dump(vectorizer, 'spam_detector_vectorizer.pkl')

    return accuracy, report





if __name__ == "__main__":
    data = load_data()
    accuracy, report = train_model(data)
    print(f"Accuracy: \n{accuracy}")
    print(f"Classification Report: \n{report}")