#!/bin/env python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

review = pd.read_csv('imdb-reviews-pt-br.csv')
review['classification'] = review['sentiment'].replace(['neg', 'pos'], [0, 1])

vectorizer = CountVectorizer(lowercase=False, max_features=50)
bag_of_words = vectorizer.fit_transform(review.text_pt)

training, testing, train_class, test_class = train_test_split(
            bag_of_words,
            review.classification,
            random_state=42
        )

logistic_regression = LogisticRegression(solver='lbfgs')
logistic_regression.fit(training, train_class)
accuracy = logistic_regression.score(testing, test_class)

print(accuracy)
