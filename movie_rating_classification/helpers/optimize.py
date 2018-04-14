from movie_rating_classification.helpers.preprocess import preprocess_data
from movie_rating_classification.helpers.training import TrainingData

import time


def optimize_for_clf(og_df, method, preprocess):
    min_vote_count_range = [0, 50, 100, 500]
    backfill_method_options = ['mean', 'median', 'mode']

    best = {
        'accuracy': 0
    }
    
    for min_vote_count in min_vote_count_range:
        for backfill_method in backfill_method_options:
            if preprocess:
                preprocess_data(
                    og_df,
                    min_vote_count,
                    backfill_method
                )
                time.sleep(5)

            training_data = TrainingData()

            accuracy = method(
                training_data.X_tr,
                training_data.X_ts,
                training_data.Y_tr,
                training_data.Y_ts
            )

            if accuracy > best['accuracy']:
                best['min_vote_count'] = min_vote_count
                best['backfill_method'] = backfill_method
                best['accuracy'] = accuracy
                print('New best: {0}\n\n'.format(best))

    return best
