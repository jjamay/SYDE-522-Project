from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy import stats

import pandas as pd


def test_mlp(x_tr, x_ts, y_tr, y_ts):
    y_tr = y_tr.values.ravel()
    y_ts = y_ts.values.ravel()

    num_neurons = x_tr.shape[1]
    num_iterations = 500

    clf = MLPClassifier(hidden_layer_sizes=(num_neurons, num_neurons, num_neurons), max_iter=num_iterations)
    clf.fit(x_tr, y_tr)

    p = clf.predict(x_ts)

    accuracy = accuracy_score(p, y_ts) * 100
    return accuracy


def test_gbc(x_tr, x_ts, y_tr, y_ts):
    gbt = GradientBoostingClassifier(max_features="log2")
    gbt.fit(x_tr, y_tr)

    p = gbt.predict(x_ts)

    accuracy = accuracy_score(p, y_ts) * 100
    return accuracy


def test_svm(x_tr, x_ts, y_tr, y_ts):

    def svc_param_selection(X, y, jobs):
        params = {'C': stats.uniform(0, 10),
                  'gamma': stats.uniform(0, 1)}

        rand_search = RandomizedSearchCV(SVC(),
                                         param_distributions=params,
                                         n_jobs=jobs,
                                         random_state=2017)
        rand_search.fit(X, y)
        print(rand_search.best_params_)
        return rand_search.best_params_

    # best_params = svc_param_selection(x_tr, y_tr, 4)

    best_params = {'C': 4.479, 'gamma': 0.1205}

    svc = SVC(kernel='linear', C=best_params['C'], gamma=best_params['gamma'])
    svc.fit(x_tr, y_tr)

    p = svc.predict(x_ts)

    accuracy = accuracy_score(p, y_ts) * 100
    return accuracy


def test_rfc(x_tr, x_ts, y_tr, y_ts):
    rft = RandomForestClassifier()
    rft.fit(x_tr, y_tr)

    p = rft.predict(x_ts)

    accuracy = accuracy_score(p, y_ts) * 100
    return accuracy


def test_logistic_regression(x_tr, x_ts, y_tr, y_ts):
    y_tr = y_tr.values.ravel()
    y_ts = y_ts.values.ravel()
    lrc = LogisticRegression()
    lrc.fit(x_tr, y_tr)

    p = lrc.predict(x_ts)

    accuracy = accuracy_score(p, y_ts) * 100
    return accuracy
