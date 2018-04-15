from movie_rating_classification.helpers.preprocess import preprocess_data
from movie_rating_classification.helpers.training import TrainingData
from movie_rating_classification.helpers.data import get_data

import time

MIN_VOTE_COUNT = 100
BACKFILL_METHOD = 'median'


def classify(method, preprocess, tune):
    if preprocess:
        preprocess_data(
            get_data(),
            MIN_VOTE_COUNT,
            BACKFILL_METHOD
        )
        time.sleep(5)

    training_data = TrainingData()

    return method(
        training_data.X_tr,
        training_data.X_ts,
        training_data.Y_tr,
        training_data.Y_ts,
        tune
    )
