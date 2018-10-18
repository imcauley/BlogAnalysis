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

    def sensitivity_test(self):
        training_file = "sensitivity_test.csv"

        losses = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'huber']
        alphas = [0.1, 0.001, 0.0001, 0.000001]
        penalty = ['l1', 'l2', 'elasticnet']

        for l in losses:
            for a in alphas:
                for p in penalty:
                    model_params = {
                        "loss": l,
                        "alpha": a,
                        "penalty": p
                    }

                    self.model = Pipeline([
                        ('vect', CountVectorizer(stop_words='english')),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(**model_params)),
                        ])

                    score = self.train_model()

                    with open(training_file, "a") as f:
                        row = [score, l, a, p]
                        row = str(row)
                        row = row[1:-1]
                        f.write(row)

    def get_data(self):
        print("Getting Data...")
        data = BlogData.get_data()
        split_data = self.seperate_data(data)

        self.data = split_data

    def seperate_data(self, data, testing_perc=0.25):
        random.shuffle(data)

        split = int(len(data) * testing_perc)

        X, Y = zip(*data)

        train_X = X[:split]
        test_X = X[split:]

        train_Y = Y[:split]
        test_Y = Y[split:]

        return train_X, train_Y, test_X, test_Y

    def train_model(self):
        train_X, train_Y, test_X, test_Y = self.data

        print("Training Model...")
        classifier = self.model.fit(train_X, train_Y)

        print("Testing Model...")
        score = classifier.score(test_X, test_Y)

        return score