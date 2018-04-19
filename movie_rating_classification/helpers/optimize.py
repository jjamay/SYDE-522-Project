from movie_rating_classification.helpers.preprocess import preprocess_data
from movie_rating_classification.helpers.training import TrainingData
from movie_rating_classification.helpers.data import get_data

import time


def optimize_for_clf(method, tune):
    min_vote_count_range = [0, 50, 100, 500]

    best = {
        'accuracy': 0
    }

    for min_vote_count in min_vote_count_range:
        preprocess_data(
            get_data(),
            min_vote_count
        )
        time.sleep(5)

        training_data = TrainingData()

        accuracy = method(
            training_data.X,
            training_data.Y,
            tune
        )

        if accuracy > best['accuracy']:
            best['min_vote_count'] = min_vote_count
            best['accuracy'] = accuracy
            print('New best: {0}\n\n'.format(best))

    return best
