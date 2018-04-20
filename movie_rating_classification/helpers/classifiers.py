from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split
)
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


def accuracy(clf, x, y, cv=5):
    """Calculates accuracy of classifier using cross-validation

    Args:
        clf (sklearn model): Classifer to evaluate
        x (np.array): Input features
        y (np.array): Classes
        cv (int, optional): Number of folds to use in cross-validation

    Returns:
        float: Mean percentage accuracy of classifier
    """
    print_classification_info(clf, x, y)
    return cross_val_score(clf, x, y, cv=cv).mean() * 100


def print_classification_info(clf, x, y):
    """Prints classification detailed results,
    including a classification report and a confusion matrix
    Note: results are only printed for one data split

    Args:
        clf (sklearn model): Classifier used to print results
        x (np.array): Input features
        y (np.array): Classes
    """
    x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, train_size=0.8, test_size=0.2)
    clf.fit(x_tr, y_tr)
    p = clf.predict(x_ts)
    print(classification_report(y_ts, p))
    print(confusion_matrix(y_ts, p))


def create_pipeline(clf):
    """Creates pipeline by scaling data before applying classifier

    Args:
        clf (sklearn model): Classifier to be used as final step in pipeline

    Returns:
        Pipeline: Final pipeline to be used to classification
    """
    return Pipeline([('scaler', MinMaxScaler()), ('clf', clf)])


def test_nb(x, y, tune):
    """Performs classification using Naive-Bayes classifier

    Args:
        x (np.array): Input features
        y (np.array): Classes
        tune (Bool): True to perform hyperparameter tuning, False otherwise

    Returns:
        Float: Mean percentage accuracy of classifier
    """
    # Perform classification without tuning
    nb = GaussianNB()
    pipeline = create_pipeline(nb)
    return accuracy(pipeline, x, y)


def test_mnb(x, y, tune):
    """Performs classification using Multinomial Naive-Bayes classifier

    Args:
        x (np.array): Input features
        y (np.array): Classes
        tune (Bool): True to perform hyperparameter tuning, False otherwise

    Returns:
        Float: Mean percentage accuracy of classifier
    """
    if tune:
        parameters = {'clf__alpha': stats.uniform(0, 1)}
        # Perform randomized searching to optimize ALPHA hyperparameter
        # using range of (0, 1) and 5 folds for cross-validation
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
        # Perform classification without tuning
        mnb = MultinomialNB(alpha=0.7886472536788195)
        pipeline = create_pipeline(mnb)
        return accuracy(pipeline, x, y)


def test_mlp(x, y, tune):
    """Performs classification using Multi-Layer Perceptron classifier

    Args:
        x (np.array): Input features
        y (np.array): Classes
        tune (Bool): True to perform hyperparameter tuning, False otherwise

    Returns:
        Float: Mean percentage accuracy of classifier
    """
    # create a neuron for each feature in the dataset
    num_neurons = x.shape[1]

    if tune:
        parameters = {
            'clf__hidden_layer_sizes': [(num_neurons, num_neurons, num_neurons), (2 * num_neurons, 2 * num_neurons, 2 * num_neurons)],
            'clf__activation': ['tanh', 'logistic', 'relu'],
            'clf__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
            'clf__max_iter': [200],
            'clf__learning_rate_init': [0.001, 0.05]
        }

        # Perform randomized searching to optimizie hidden layer sizes, the activation function
        # alpha, the maximum number of iterations to perform, and the initial learning rate
        # using 5 fold cross validation
        rand_search = RandomizedSearchCV(
            estimator=create_pipeline(MLPClassifier(verbose=10)),
            param_distributions=parameters,
            n_jobs=-1,
            cv=5,
        )

        rand_search.fit(x, y)
        best_params = rand_search.best_params_
        print("best_params: {0}".format(best_params))
        return rand_search.best_score_ * 100
    else:
        # Perform classification without tuning
        clf = MLPClassifier(
            hidden_layer_sizes=(2 * num_neurons, 2 * num_neurons, 2 * num_neurons),
            max_iter=200,
            activation='tanh',
            learning_rate_init=0.001,
            alpha=0.001
        )

        pipeline = create_pipeline(clf)
        return accuracy(pipeline, x, y)


def test_gbc(x, y, tune):
    """Performs classification using Gradient Boost classifier

    Args:
        x (np.array): Input features
        y (np.array): Classes
        tune (Bool): True to perform hyperparameter tuning, False otherwise

    Returns:
        Float: Mean percentage accuracy of classifier
    """
    # Perform classification without tuning. It was determined through trial-and-error
    # that log2 features produced the highest accuracy
    gbt = GradientBoostingClassifier(max_features="log2")
    pipeline = create_pipeline(gbt)
    return accuracy(pipeline, x, y)


def test_svm(x, y, tune):
    """Performs classification using SVM classifier

    Args:
        x (np.array): Input features
        y (np.array): Classes
        tune (Bool): True to perform hyperparameter tuning, False otherwise

    Returns:
        Float: Mean percentage accuracy of classifier
    """
    if tune:
        params = {'clf__C': stats.uniform(0, 100),
                  'clf__gamma': stats.uniform(0, 1)}

        # Perform hyperparameter tuning on C and gamma using randomized searching
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
        # Perform classification without tuning
        svc = SVC(gamma=0.93121875826256051, C=70.331975797816796)
        pipeline = create_pipeline(svc)
        return accuracy(pipeline, x, y)


def test_logistic_regression(x, y, tune):
    """Performs classification using Logistic Regression classifier

    Args:
        x (np.array): Input features
        y (np.array): Classes
        tune (Bool): True to perform hyperparameter tuning, False otherwise

    Returns:
        Float: Mean percentage accuracy of classifier
    """
    # Perform classification without tuning
    lrc = LogisticRegression()
    pipeline = create_pipeline(lrc)
    return accuracy(pipeline, x, y)
