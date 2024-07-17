import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
# print(sklearn.__version__)

def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    data = pd.read_csv(url, sep='\t', header=None, names=['label','text' ])
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data


def train_model(data):
    x = data['text']
    y = data['label']





if __name__ == "__main__":
    data = load_data()
    print(data)