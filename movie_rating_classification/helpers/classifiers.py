from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import (RandomizedSearchCV, cross_val_score)
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.preprocessing import (StandardScaler, MinMaxScaler)
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


def accuracy(clf, x, y, cv=5):
    return cross_val_score(clf, x, y, cv=cv).mean() * 100


def create_pipeline(clf):
    return Pipeline([('scaler', MinMaxScaler()), ('clf', clf)])


def test_nb(x, y, tune):
    nb = GaussianNB()
    pipeline = create_pipeline(nb)
    return accuracy(pipeline, x, y)


def test_mnb(x, y, tune):
    if tune:
        parameters = {'clf__alpha': stats.uniform(0, 1)}
        rand_search = RandomizedSearchCV(
            create_pipeline(MultinomialNB()),
            param_distributions=parameters,
            n_jobs=-1,
            cv=5)
        rand_search.fit(x, y)
        best_params = rand_search.best_params_
        print("best_params: {0}".format(best_params))
        return rand_search.best_score_ * 100
    else:
        mnb = MultinomialNB(alpha=0.19383194558775096)
        pipeline = create_pipeline(mnb)
        return accuracy(pipeline, x, y)


def test_mlp(x, y, tune):
    num_iterations = 500
    num_neurons = x.shape[1]

    if tune:
        parameters = {
            'clf__hidden_layer_sizes': [(num_neurons, num_neurons, num_neurons), (2*num_neurons, 2*num_neurons, 2*num_neurons)],
            'clf__activation': ['tanh', 'logistic', 'relu'],
            'clf__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
            'clf__max_iter': [200],
            'clf__learning_rate_init': [0.001, 0.05]
        }

        rand_search = RandomizedSearchCV(
            estimator=create_pipeline(MLPClassifier(verbose=10)),
            param_distributions=parameters,
            n_jobs=-1,
            cv=5
        )

        rand_search.fit(x, y)
        best_params = rand_search.best_params_
        print("best_params: {0}".format(best_params))
        return rand_search.best_score_ * 100
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(num_neurons, num_neurons, num_neurons),
            max_iter=num_iterations
        )
        pipeline = create_pipeline(clf)
        return accuracy(pipeline, x, y)


def test_gbc(x, y, tune):
    gbt = GradientBoostingClassifier(max_features="log2")
    pipeline = create_pipeline(gbt)
    return accuracy(pipeline, x, y)


def test_svm(x, y, tune):
    C = 40.82342548346405
    GAMMA = 0.03839085006161691

    if tune:
        params = {'clf__C': stats.uniform(0, 100),
                  'clf__gamma': stats.uniform(0, 1)}

        rand_search = RandomizedSearchCV(create_pipeline(SVC()),
                                         n_iter=20,
                                         param_distributions=params,
                                         n_jobs=-1,
                                         cv=5,
                                         random_state=2017)
        rand_search.fit(x, y)
        print("best_params: {0}".format(rand_search.best_params_))
        return rand_search.best_score_ * 100
    else:
        svc = SVC(C=C, gamma=GAMMA)
        pipeline = create_pipeline(svc)
        return accuracy(pipeline, x, y)


def test_rfc(x, y, tune):
    rft = RandomForestClassifier()
    pipeline = create_pipeline(rft)
    return accuracy(pipeline, x, y)


def test_logistic_regression(x, y, tune):
    lrc = LogisticRegression()
    pipeline = create_pipeline(lrc)
    return accuracy(pipeline, x, y)
