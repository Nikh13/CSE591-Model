import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import os.path as path

__author__ = 'Nikhil'

df = pd.read_csv('clustered_1000.csv', encoding='utf-8', nrows=100)   # columns names if no header
vect = TfidfVectorizer()
X = vect.fit_transform(df['tags'])
y = df['cluster_id']

clf = MultinomialNB()
if path.exists('model.pkl'):
    clf = joblib.load('model.pkl')
else:
    clf.fit(X, y)
    joblib.dump(clf, 'model.pkl')

print(clf.classes_)
docs_new = ['switzerland']
X_new_tfidf = vect.transform(docs_new)
print(clf.predict_proba(X_new_tfidf))
