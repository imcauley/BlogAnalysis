import random

import BlogData

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class TextClassifier:
    def __init__(self):
        self.model = Pipeline([
            ('vect', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier()),
            ])

    def get_data(self):
        data = BlogData.get_data()
        return data

    def seperate_data(data, testing_perc=0.25):
        random.shuffle(data)

        split = int(len(data) * testing_perc)

        X, Y = zip(*data)

        train_X = X[:split]
        test_X = X[split:]

        train_Y = Y[:split]
        test_Y = Y[split:]

        return train_X, train_Y, test_X, test_Y

    def train_model(self):
        pass