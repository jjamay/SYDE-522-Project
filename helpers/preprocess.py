import ast
import numpy as np
import string
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

CREW_ATTRIBUTES = ['cast', 'director', 'production_companies', 'producers', 'executive_producers']


# def is_all_ascii(chars):
#     if type(chars) == type([]):
#         chars = list_of_str_to_str(chars)

#     if pd.isnull(chars):
#         return True

#     printable = set(string.printable)
#     for char in chars:
#         if char not in printable:
#             return False

#     return True


def list_of_str_to_str(list_of_str):
    list_of_str = get_literal_eval(list_of_str)

    # remove non unicode words
    for word in list_of_str:
        word = word.replace(u'\xa0', u' ')
        
        try:
            word.encode()
        except:
            print(word)
            list_of_str.remove(word)
            print(list_of_str)

    return " ".join(list_of_str)


def get_role_list(people, role):
    people = ast.literal_eval(people)
    crew = []

    for person in people:
        if person['job'] == role:
            crew.append(str(person['name']))

    return crew if len(crew) else []


# def remove_rows_without_revenue_cost(df):
#     # returns a pandas dataframe
#     return df[np.isfinite(df['revenue_divide_budget'])]


# def remove_rows_with_non_english_movies(df):
#     # returns a pandas dataframe
#     df = df[df['original_language'] == 'en']
#     df = df.drop(['original_language'], 1)
#     return df

def bin_ratings(df):
    df['rating'] = df['vote_average'].apply(lambda x: int(round(x * 0.4)))
    return df


def binarize_english(df):
    df['is_english'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)
    return df


def binarize_homepage(df):
    df['homepage'] = df['homepage'].apply(lambda x: 0 if x == np.nan else 1)
    return df


def binarize_belongs_to_collection(df):
    df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x: 0 if x == np.nan else 1)
    return df


def add_producers_feature(df):
    df['producers'] = df['crew'].apply(get_role_list, role='Producer')
    return df


def add_executive_producers_feature(df):
    df['executive_producers'] = df['crew'].apply(get_role_list, role='Executive Producer')
    return df


def binarize_genres(df):
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if x != np.nan else [])

    genres = [
        'Drama',
        'Comedy',
        'Thriller',
        'Action',
        'Romance',
        'Adventure',
        'Crime',
        'Science Fiction',
        'Horror',
        'Family',
        'Fantasy',
        'Mystery',
        'Animation',
        'History',
        'Music',
        'War',
        'Western',
        'Documentary',
        'Foreign'
    ]

    for genre in genres:
        df['is_{}'.format(genre.lower().replace(' ', '_'))] = df['genres'].apply(lambda x: 1 if genre in x else 0)
    return df


# def remove_rows_with_non_ascii(df):
#     text_cols = [
#         # 'genres',
#         'overview',
#         'production_companies',
#         'production_countries',
#         'tagline',
#         'title',
#         'cast',
#         'keywords',
#         'director',
#         'producers',
#         'executive_producers',
#         'belongs_to_collection'
#     ]

#     for col in text_cols:
#         df = df[df[col].apply(is_all_ascii)]

#     return df


def generate_name_key(str):
    '''
    examples of str: Tom Hanks, Michael, Michael Buble Test
    '''

    return str.decode('utf-8', errors='replace').lower().replace(' ', '_')


def get_literal_eval(data):
    '''
    probably could use a better name
    returns literal_eval or returns original data in list form (for director)
    '''

    if type(data) != type([]):
        try:
            return ast.literal_eval(data)
        except:
            return [data] if data is not np.nan else []

    return data


def get_avg_scores_for_attribute(df, attribute):
    ratings = {}
    min_vote_count = 1000

    movies = df[df['vote_count'] > min_vote_count]

    for index, row in movies.iterrows():
        group = row[attribute]

        # handle director case that isn't wrapped by []
        group = get_literal_eval(group)

        for item in group:
            item_key = generate_name_key(item)

            if item_key not in ratings:
                ratings[item_key] = {
                    'num_movies': 1,
                    'total_score': row['vote_average'],
                    'avg_score': row['vote_average']
                }
            else:
                ratings[item_key]['num_movies'] += 1
                ratings[item_key]['total_score'] += row['vote_average']
                ratings[item_key]['avg_score'] = ratings[item_key]['total_score'] / ratings[item_key]['num_movies']

    return ratings


def calculate_total_score(data, ratings):
    '''
    calculates total score for actors, production_companies, etc.
    '''

    total_score = 0

    for x in data:
        x_key = generate_name_key(x)
        total_score += ratings[x_key]['avg_score'] if x_key in ratings else 0

    return total_score


def calculate_final_production_score(row, ratings):
    scores = {}

    for x in CREW_ATTRIBUTES:
        data = row[x]

        # handle director case that isn't wrapped by []
        data = get_literal_eval(data)

        total_score = calculate_total_score(data, ratings[x])

        # more famous actors usually means higher review (e.g. Ocean's 11)
        if x == 'cast':
            scores[x] = total_score
        elif len(data):
            scores[x] = (total_score / len(data))

    final_score = sum(scores.values())

    return final_score


def get_movie_scores(df):
    ratings = {}

    for x in CREW_ATTRIBUTES:
        ratings[x] = get_avg_scores_for_attribute(df, x)

    df['production_score'] = df.apply(calculate_final_production_score, ratings=ratings, axis=1)
    return df


def parse_production_countries(df):
    df['production_countries'] = df['production_countries'].apply(lambda x: get_literal_eval(x))
    return df


def convert_keywords_to_string(df):
    df['keywords'] = df['keywords'].apply(list_of_str_to_str)
    return df


def drop_unnecessary_columns(df):
    df = df.drop([
        'id',
        'imdb_id',
        'poster_path',
        'video',
        'status',
        'weighted_rating',
        'original_title',
        'crew',
        'producers',
        'executive_producers',
        'cast',
        'director',
        'production_companies',
        'genres',
        'original_language',
        'revenue',
        'vote_count',
        'adult',
        'release_date',  # ADD BACK IN WHEN READY
        'overview',
        'title',
        'tagline',
        'vote_average'
    ], 1)
    return df


def preprocess_data(df):
    # note that order matters!
    # df = remove_rows_without_revenue_cost(df)
    # df = remove_rows_with_non_english_movies(df)
    df = binarize_homepage(df)
    df = add_producers_feature(df)
    df = add_executive_producers_feature(df)
    # df = remove_rows_with_non_ascii(df)
    df = get_movie_scores(df)
    df = binarize_english(df)
    df = bin_ratings(df)
    df = binarize_genres(df)
    df = binarize_belongs_to_collection(df)
    df = parse_production_countries(df)
    df = convert_keywords_to_string(df)
    df = drop_unnecessary_columns(df)
    return df
