from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN

import collections
import pandas as pd
import numpy as np


X_FILE = r'../dataset/X.csv'
Y_FILE = r'../dataset/Y.csv'


class TrainingData:

    def __init__(self):
        self.X = pd.read_csv(X_FILE)
        self.Y = pd.read_csv(Y_FILE)
        self.Y = np.reshape(self.Y.values, [self.Y.shape[0], ])

        # Oversample minority classes using Synthetic Minority Over-sampling Technique
        # Undersample majority classes using Edited Nearest Neighbour method
        sm = SMOTEENN()
        self.X, self.Y = sm.fit_sample(self.X, self.Y)

        print(collections.Counter(self.Y))

        # Encode the four rating classes (terrible, poor, average, excellent)
        # into integer values for testing
        label_enc = LabelEncoder()
        self.Y = label_enc.fit_transform(self.Y)

        """ COLUMNS USED IN TRAINING
        'belongs_to_collection'
        'budget'
        'homepage'
        'runtime'
        'spoken_languages'
        'cast_size'
        'crew_size'
        'production_score'
        'is_english'
        'is_drama'
        'is_comedy'
        'is_thriller'
        'is_action'
        'is_romance'
        'is_adventure'
        'is_crime'
        'is_science_fiction'
        'is_horror'
        'is_family'
        'is_fantasy'
        'is_mystery'
        'is_animation'
        'is_history'
        'is_music'
        'is_war'
        'is_western'
        'is_documentary'
        'is_foreign'
        'is_tv_movie'
        'prod_uk'
        'prod_usa'
        'prod_france'
        'prod_other'
        """
