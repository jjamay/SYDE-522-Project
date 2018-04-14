from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats


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
