import pandas as pd

MOVIES_MD = r'../dataset/movies_tmdbMeta.csv'


def get_data():
    return pd.read_csv(MOVIES_MD)
