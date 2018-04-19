from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np


X_FILE = r'../dataset/X.csv'
Y_FILE = r'../dataset/Y.csv'


class TrainingData:

    def __init__(self):
        self.X = pd.read_csv(X_FILE)
        self.X.fillna('', inplace=True)  # can't have nan in any of the columns
        self.Y = pd.read_csv(Y_FILE)
        self.Y = np.reshape(self.Y.values, [self.Y.shape[0], ])

        self.X = self.generate_features(self.X)

        # pca = PCA(n_components=35)
        # self.X = pca.fit_transform(self.X)

        label_enc = LabelEncoder()
        self.Y = label_enc.fit_transform(self.Y)

    def generate_features(self, data):
        mapper = DataFrameMapper([
            ('belongs_to_collection', None),
            # ('revenue_divide_budget', None),
            ('budget', None),
            ('homepage', None),
            ('popularity', None),
            ('runtime', None),
            ('spoken_languages', None),
            # ('keywords', TfidfVectorizer()),
            ('cast_size', None),
            ('crew_size', None),
            ('production_score', None),
        #    ('release_date', None),
            ('is_english', None),
            ('is_drama', None),
            ('is_comedy', None),
            ('is_thriller', None),
            ('is_action', None),
            ('is_romance', None),
            ('is_adventure', None),
            ('is_crime', None),
            ('is_science_fiction', None),
            ('is_horror', None),
            ('is_family', None),
            ('is_fantasy', None),
            ('is_mystery', None),
            ('is_animation', None),
            ('is_history', None),
            ('is_music', None),
            ('is_war', None),
            ('is_western', None),
            ('is_documentary', None),
            ('is_foreign', None),
            ('is_tv_movie', None),
            ('prod_uk', None),
            ('prod_usa', None),
            ('prod_france', None),
            ('prod_other', None),
            # ('vote_count', None)
        ], input_df=True)

        return mapper.fit_transform(data)
