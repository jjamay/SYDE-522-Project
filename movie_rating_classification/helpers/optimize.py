from movie_rating_classification.helpers.preprocess import preprocess_data
from movie_rating_classification.helpers.training import TrainingData


def optimize_for_clf(og_df, method):
    min_vote_count_range = [100, 500, 1000, 5000]
    backfill_method_options = ['mean', 'median', 'mode']
    num_vectorizer_features_range = [5, 10, 20]

    best = {
        'accuracy': 0
    }

    for min_vote_count in min_vote_count_range:
        for backfill_method in backfill_method_options:
            df = preprocess_data(
                og_df,
                min_vote_count,
                backfill_method
            )

            y = df[['rating']]
            x = df.drop(['rating'], 1)

            for num_vectorizer_features in num_vectorizer_features_range:
                training_data = TrainingData(
                    X_df=x,
                    Y_df=y,
                    num_vectorizer_features=num_vectorizer_features
                )

                accuracy = method(
                    training_data.X_tr,
                    training_data.X_ts,
                    training_data.Y_tr,
                    training_data.Y_ts
                )

                if accuracy > best['accuracy']:
                    best['min_vote_count'] = min_vote_count
                    best['backfill_method'] = backfill_method
                    best['num_vectorizer_features'] = num_vectorizer_features                
                    best['accuracy'] = accuracy
                    print('New best: {0}\n\n'.format(best))

    return best