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
import pandas as pd

MOVIES_MD = r'../dataset/movies_tmdbMeta.csv'


def run(classifier, preprocess, optimize):
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
        print 'Error: Invalid classifier specified'
        sys.exit(2)
    og_df = pd.read_csv(MOVIES_MD)
    if optimize:
        best = optimize_for_clf(og_df, method, preprocess)
        print('Best performance with {0}: {1}'.format(classifier, best))
    else:
        accuracy = classify(og_df, method, preprocess)
        print('Accuracy with {0}: {1}'.format(classifier, accuracy))


def main(argv):
    classifier = None
    preprocess = False
    optimize = False

    try:
        opts, args = getopt.getopt(argv, "hpoc:")
    except getopt.GetoptError:
        print 'run.py -c <classifier> -p -o'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'run.py -c <classifier> -p -o'
            sys.exit()
        if opt == '-c':
            classifier = arg
        if opt == '-p':
            preprocess = True
        if opt == '-o':
            optimize = True

    run(classifier, preprocess, optimize)


if __name__ == "__main__":
    main(sys.argv[1:])
