from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import numpy as np


X_FILE = r'dataset/X.csv'
Y_FILE = r'dataset/Y.csv'

N = 20
TRAIN_SIZE = 0.8

class TrainingData:

    def __init__(
        self,
        X_df=None,
        Y_df=None,
    ):
        self.X = pd.read_csv(X_FILE) if X_df is None else X_df
        self.X.fillna('', inplace=True) # can't have nan in any of the columns
        self.Y = pd.read_csv(Y_FILE) if Y_df is None else Y_df

        if Y_df is None:
            self.Y = np.reshape(self.Y.values, [self.Y.shape[0],])
        
        self.features = self.generate_features(self.X)
        self.X_tr, self.X_ts, self.Y_tr, self.Y_ts = train_test_split(self.features, self.Y, train_size=TRAIN_SIZE)
        label_enc = LabelEncoder()
        self.Y_tr = label_enc.fit_transform(self.Y_tr);
        self.Y_ts = label_enc.transform(self.Y_ts);
        
    def generate_features(self, data):
        mapper = DataFrameMapper([
            ('belongs_to_collection', None),
            ('revenue_divide_budget', MinMaxScaler()),
            ('homepage', None),
            ('popularity', MinMaxScaler()),
            ('runtime', MinMaxScaler()),
            ('spoken_languages', MinMaxScaler()),
#             ('keywords', HashingVectorizer(n_features=N)),
            ('cast_size', MinMaxScaler()),
            ('crew_size', MinMaxScaler()),
            ('production_score', MinMaxScaler()),
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
            ('prod_other', None)
        ], input_df=True)


        return mapper.fit_transform(data)
