from sklearn.metrics import accuracy_score
from movie_rating_classification.helpers.optimize import optimize_for_clf
from sklearn.neural_network import MLPClassifier

import pandas as pd


def test_rfc(x_tr, x_ts, y_tr, y_ts):
    y_tr = y_tr.values.ravel()
    y_ts = y_ts.values.ravel()
    
    num_neurons = 10#x_tr.shape[1]
    num_iterations = 5000

    clf = MLPClassifier(hidden_layer_sizes=(num_neurons, num_neurons, num_neurons), max_iter=num_iterations)
    clf.fit(x_tr, y_tr)

    p = clf.predict(x_ts)

    accuracy = accuracy_score(p, y_ts) * 100
    return accuracy


movies_md = r'../../dataset/movies_tmdbMeta.csv'
og_movies_md_df = pd.read_csv(movies_md)
best = optimize_for_clf(og_movies_md_df, test_rfc)
print('Best performance with rfc: {0}'.format(best))
