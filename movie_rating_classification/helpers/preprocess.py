import ast
import numpy as np
import pandas as pd


pd.options.mode.chained_assignment = None  # default='warn'

CREW_ATTRIBUTES = ['cast', 'director', 'production_companies', 'producers', 'executive_producers']


def get_role_list(people, role):
    """Gets a list of all people involved in a movie who's
    job is a certain role

    Args:
        people (String): String literal of all people in a movie crew
        role (String): Role

    Returns:
        List: list of people whose job is the specified role
    """
    people = ast.literal_eval(people)
    crew = []

    for person in people:
        if person['job'] == role:
            crew.append(str(person['name']))

    return crew if len(crew) else []


def remove_rows_without_feature(df, feature):
    """Removes all rows with missing value for feature

    Args:
        df (DataFrame): Pandas dataframe
        feature (String): Feature name

    Returns:
        DataFrame: Modified dataframe
    """
    return df[np.isfinite(df[feature])]


def remove_rows_with_non_english_movies(df):
    """Removes all rows with english as the original language

    Args:
        df (DataFrame): Pandas dataframe

    Returns:
        DataFrame: Modified dataframe
    """
    df = df[df['original_language'] == 'en']
    return df


def bin_ratings(df):
    """Bins movie ratings into one of four categories:
    terrible, poor, average, and excellent

    Args:
        df (DataFrame): Pandas dataframe

    Returns:
        DataFrame: Modified dataframe
    """
    def bin(val):
        if val < 2.5:
            return 'terrible'
        if val < 5:
            return 'poor'
        if val < 7.5:
            return 'average'
        return 'excellent'

    df['rating'] = df['vote_average'].apply(bin)
    return df


def binarize_english(df):
    """Performs binarization of original language
    1 if english, 0 if not

     Args:
        df (DataFrame): Pandas dataframe

    Returns:
        DataFrame: Modified dataframe
    """
    df['is_english'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)
    return df


def binarize_homepage(df):
    """Performs binarization of homepage
    1 if there is a homepage, 0 if not

    Args:
        df (DataFrame): Pandas dataframe

    Returns:
        DataFrame: Modified dataframe
    """
    df['homepage'] = df['homepage'].apply(lambda x: 0 if pd.isnull(x) else 1)
    return df


def binarize_belongs_to_collection(df):
    """Performs binarization of belongs_to_collection
    1 if movie belonds to a collection, 0 if not

    Args:
        df (DataFrame): Pandas dataframe

    Returns:
        DataFrame: Modified dataframe
    """
    df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x: 0 if pd.isnull(x) else 1)
    return df


def add_producers_feature(df):
    """Extracts all the producers from a movie crew list
    and adds a new column containing a list of them

    Args:
        df (DataFrame): Pandas dataframe

    Returns:
        DataFrame: Modified dataframe
    """
    df['producers'] = df['crew'].apply(get_role_list, role='Producer')
    return df


def add_executive_producers_feature(df):
    """Extracts all the executive producers from a movie crew list
    and adds a new column containing a list of them

    Args:
        df (DataFrame): Pandas dataframe

    Returns:
        DataFrame: Modified dataframe
    """
    df['executive_producers'] = df['crew'].apply(get_role_list, role='Executive Producer')
    return df


def binarize_genres(df):
    """Performs Multi-label binarization on the genre column

     Args:
        df (DataFrame): Pandas dataframe

    Returns:
        DataFrame: Modified dataframe
    """
    df['genres'] = df['genres'].apply(lambda x: [] if pd.isnull(x) else ast.literal_eval(x))

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
        'Foreign',
        'TV Movie'
    ]

    for genre in genres:
        df['is_{}'.format(genre.lower().replace(' ', '_'))] = df['genres'].apply(lambda x: 1 if genre in x else 0)
    return df


def generate_name_key(str):
    """Helper function to generate a unique key for a string

    Args:
        str (String): string to generate key for

    Returns:
        String: Unique key
    """
    return str.decode('utf-8', errors='replace').lower().replace(' ', '_')


def get_literal_eval(data):
    """Returns evaluation of string literal
    of original data in list format
    Args:
        data (String): String literal to evaluate

    Returns:
        List(String): List evaluated from string
    """
    if type(data) != type([]):
        try:
            return ast.literal_eval(data)
        except:
            return [data] if data is not np.nan else []

    return data


def get_avg_scores_for_attribute(df, attribute, min_vote_count):
    """

    Args:
        df (DataFrame): Pandas dataframe
        attribute (String): Attribute to calculate scores for
        min_vote_count (Integer): Movies must have more than
                                  this number of votes for their
                                  rating to be included in the calculation

    Returns:
        DataFrame: Modified dataframe
    """
    ratings = {}

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
    """Summary
    Calculates the total score for a specific attribute

    Args:
        data (array of strings): contains list of names (actors, directors, producers, or executive producers)
        ratings (dictionary): dictionary that maps an actor to its rating

    Returns:
        float: total score
    """
    total_score = 0

    for x in data:
        x_key = generate_name_key(x)
        total_score += ratings[x_key]['avg_score'] if x_key in ratings else 0

    return total_score


def calculate_final_production_score(row, ratings):
    """Summary
    for every actor, producer, executive producer and director who is in a movie with more than min_vote_count, 
    we take the average of the rating of every movie that the person participated in and generate maps {producer:rating}, 
    {director:rating} etc.

    for every movie, we calculate a production score which is equal to 
    sum(actor ratings) + avg(director ratings) + avg(producer ratings) + avg(executive producer ratings)

    we take the sum of actors because generally the number of popular actors involved in a movie is directly related to how good a movie is (think Ocean's 11)

    Args:
        row (dataframe): row in the dataframe
        ratings (dictionary): dictionary that maps an actor to its rating

    Returns:
        float: final score
    """
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


def get_movie_scores(df, min_vote_count=1000):
    """Calculates a production score for each movie based on a number
    of movie attributes

    Args:
        df (DataFrame): Pandas dataframe
        min_vote_count (Integer): Movies must have more than
                                  this number of votes for their
                                  rating to be included in the calculation

    Returns:
        DataFrame: Modified dataframe
    """
    ratings = {}

    for x in CREW_ATTRIBUTES:
        ratings[x] = get_avg_scores_for_attribute(df, x, min_vote_count)

    df['production_score'] = df.apply(calculate_final_production_score, ratings=ratings, axis=1)
    return df


def binarize_production_countries(df):
    """Performs multi-label binarization for production_countries

    Args:
        df (DataFrame): Pandas dataframe

    Returns:
        DataFrame: Modified dataframe
    """
    df['production_countries'] = df['production_countries'].apply(lambda x: get_literal_eval(x))
    countries = {
        'United States of America': 'usa',
        'United Kingdom': 'uk',
        'France': 'france'
    }

    for country, short in countries.iteritems():
        df['prod_{}'.format(short)] = df['production_countries'].apply(lambda x: 1 if country in x else 0)

    def check_other(prod_countries):
        for c in prod_countries:
            if c not in countries:
                return 1
        return 0

    df['prod_other'] = df['production_countries'].apply(check_other)

    return df


def drop_unnecessary_columns(df):
    """Drops all columns not needed for training

    Args:
        df (DataFrame): Pandas dataframe

    Returns:
        DataFrame: Modified dataframe
    """
    df = df.drop([
        'id',
        'imdb_id',
        'poster_path',
        'video',
        'status',
        'weighted_rating',  # Only average_rating was used for this project
        'original_title',
        'crew',  # Used in production_score
        'producers',  # Used in production_score
        'executive_producers',  # Used in production_score
        'cast',  # Used in production_score
        'director',  # Used in production_score
        'production_companies',  # Used in production_score
        'production_countries',  # Binarized
        'genres',  # Binarized
        'original_language',  # Binarized
        'adult',  # No adult movies in the dataset, so no variance between movies
        'release_date',  # Not being considered for this project
        'overview',
        'title',
        'tagline',
        'vote_average',  # Ratings have been binned
        'popularity',  # Only considering average_rating
        'vote_count',  # We are making a predictor, so it makes no sense to use vote counts as input
        'revenue',  # We are making a predictor, so it makes no sense to use revenue as input
        'keywords',  # Not considering keywords for this project
        'revenue_divide_budget',  # We are making a predictor, so it makes no sense to use revenue/budget as input
    ], 1)
    return df


def preprocess_data(df, min_vote_count=1000):
    """Performs all data preprocessing and exports input and output
    csv files

    Args:
        df (DataFrame): Pandas dataframe
    """
    # note that order matters!
    df = remove_rows_without_feature(df, 'budget')
    df = remove_rows_without_feature(df, 'runtime')
    df = remove_rows_with_non_english_movies(df)
    df = binarize_homepage(df)
    df = add_producers_feature(df)
    df = add_executive_producers_feature(df)
    df = get_movie_scores(df, min_vote_count)
    df = binarize_english(df)
    df = bin_ratings(df)
    df = binarize_genres(df)
    df = binarize_belongs_to_collection(df)
    df = binarize_production_countries(df)
    df = drop_unnecessary_columns(df)

    # Export to CSV
    y = df[['rating']]
    x = df.drop(['rating'], 1)

    y.to_csv(r'../dataset/Y.csv', index=False)
    x.to_csv(r'../dataset/X.csv', index=False)
