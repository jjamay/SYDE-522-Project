from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline


def accuracy(classifier, x, y, cv=5):
    clf = make_pipeline(MinMaxScaler(), classifier)
    return cross_val_score(clf, x, y, cv=cv).mean() * 100


def test_mlp(x, y, tune):
    num_neurons = x.shape[1]
    num_iterations = 500

    clf = MLPClassifier(hidden_layer_sizes=(num_neurons, num_neurons, num_neurons), max_iter=num_iterations)
    return accuracy(clf, x, y)


def test_gbc(x, y, tune):
    gbt = GradientBoostingClassifier(max_features="log2")
    return accuracy(gbt, x, y)


def test_svm(x, y, tune):
    C = 40.82342548346405
    GAMMA = 0.03839085006161691

    if tune:
        def svc_param_selection(X, y, jobs):
            params = {'C': stats.uniform(0, 100),
                      'gamma': stats.uniform(0, 1)}

            rand_search = RandomizedSearchCV(SVC(),
                                             n_iter=20,
                                             param_distributions=params,
                                             n_jobs=jobs,
                                             random_state=2017)
            rand_search.fit(X, y)
            print(rand_search.best_params_)
            return rand_search.best_params_

        best_params = svc_param_selection(x, y, 4)

        svc = SVC(C=best_params['C'], gamma=best_params['gamma'])
    else:
        svc = SVC(C=C, gamma=GAMMA)

    return accuracy(svc, x, y)


def test_rfc(x, y, tune):
    rft = RandomForestClassifier()
    return accuracy(rft, x, y)


def test_logistic_regression(x, y, tune):
    lrc = LogisticRegression()
    return accuracy(lrc, x, y)
