from sklearn.metrics import accuracy_score
from movie_rating_classification.helpers.optimize import optimize_for_clf
from sklearn.linear_model import LogisticRegression

import pandas as pd


def test_logistic_regression(x_tr, x_ts, y_tr, y_ts):
    y_tr = y_tr.values.ravel()
    y_ts = y_ts.values.ravel()
    lrc = LogisticRegression()
    lrc.fit(x_tr, y_tr)

    p = lrc.predict(x_ts)

    accuracy = accuracy_score(p, y_ts) * 100
    return accuracy


movies_md = r'../../dataset/movies_tmdbMeta.csv'
og_movies_md_df = pd.read_csv(movies_md)
best = optimize_for_clf(og_movies_md_df, test_logistic_regression)
print('Best performance with gbc: {0}'.format(best))
