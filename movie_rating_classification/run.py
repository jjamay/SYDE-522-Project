from movie_rating_classification.helpers.classifiers import (
    test_mlp,
    test_rfc,
    test_gbc,
    test_svm,
    test_logistic_regression
)
from movie_rating_classification.helpers.optimize import optimize_for_clf
from movie_rating_classification.helpers.classify import classify

import sys
import getopt


def run(classifier, preprocess, optimize, tune):
    if classifier == "svm":
        method = test_svm
    elif classifier == "rfc":
        method = test_rfc
    elif classifier == "gbc":
        method = test_gbc
    elif classifier == 'mlp':
        method = test_mlp
    elif classifier == 'lr':
        method = test_logistic_regression
    else:
        print('Error: Invalid classifier specified')
        sys.exit(2)

    if optimize:
        best = optimize_for_clf(method, tune)
        print('Best performance with {0}: {1}'.format(classifier, best))
    else:
        accuracy = classify(method, preprocess, tune)
        print('Accuracy with {0}: {1}'.format(classifier, accuracy))


def main(argv):
    classifier = None
    preprocess = False
    optimize = False
    tune = False

    classifier = argv[0]

    opts, args = getopt.getopt(argv[1:], "hpotc:")

    for opt, arg in opts:
        if opt == '-h':
            print('run.py <classifier> -p -o -t')
            sys.exit()
        if opt == '-p':
            preprocess = True
        if opt == '-o':
            optimize = True
        if opt == '-t':
            tune = True
            
    run(classifier, preprocess, optimize, tune)


if __name__ == "__main__":
    main(sys.argv[1:])
