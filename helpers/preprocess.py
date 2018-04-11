import ast
import numpy as np
import string
import pandas as pd 


def is_all_ascii(chars):
    if pd.isnull(chars):
        return True
    
    printable = set(string.printable)
    for char in chars:
        if char not in printable:
            return False

    return True

def list_of_str_to_str(list_of_str):
    return " ".join(list_of_str) if type(list_of_str) == list else list_of_str
    
def get_role_list(people, role):
    people = ast.literal_eval(people)
    crew = []
    
    for person in people:
        if person['job'] == role:
            crew.append(person['name'])
    
    return crew if len(crew) else np.nan

def remove_rows_without_revenue_cost(df):
	# returns a pandas dataframe
	return df[np.isfinite(df['revenue_divide_budget'])]

def remove_rows_with_non_english_movies(df):
	# returns a pandas dataframe
	df = df[df['original_language'] == 'en']
	df = df.drop(['original_language'], 1)
	return df

def binarize_homepage(df):
	df['homepage'] = df['homepage'].apply(lambda x: 0 if x == np.nan else 1)
	return df

def add_producers_feature(df):
	df['producers'] = df['crew'].apply(get_role_list, role='Producer')
	return df

def add_executive_producers_feature(df):
	df['executive_producers'] = df['crew'].apply(get_role_list, role='Executive Producer')
	return df

def convert_columns_with_list_of_str_to_str(df):
	columns_with_list_of_str = [
	    'genres', 
	    'production_countries', 
	    'production_companies', 
	    'cast', 
	    'keywords',
	]

	for col in columns_with_list_of_str:
	    df[col] = df[col].apply(lambda x: list_of_str_to_str(ast.literal_eval(x)) if x != np.nan else x)

	return df

def remove_rows_with_non_ascii(df):
	text_cols = [
	    'genres',
	    'overview',
	    'production_companies',
	    'production_countries',
	    'tagline',
	    'title',
	    'cast',
	    'keywords',
	    'director',
	    'producers',
	    'executive_producers',
	    'belongs_to_collection'
	]

	for col in text_cols:
	    df = df[df[col].apply(is_all_ascii)]

	return df

def get_avg_scores(df, component):
    ratings = {}
    min_vote_count = 1000

    movies = df[df['vote_count'] > min_vote_count]

    for index, row in movies.iterrows():
        group = row[component]
        
        # handle director case that isn't wrapped by []
        try:
            group = ast.literal_eval(group)
        except:
            group = [group]
        
        for item in group:
            item_key = item.lower().replace(' ', '_')

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

def get_movie_scores(df):
	actor_ratings = get_avg_scores(df, 'cast')
	director_ratings = get_avg_scores(df, 'director')
	production_company_ratings = get_avg_scores(df, 'production_companies')
	producer_ratings = get_avg_scores(df, 'producers')
	executive_producers_ratings = get_avg_scores(df, 'executive_producers')

def preprocess_data(df):
	df = remove_rows_without_revenue_cost(df)
	df = remove_rows_with_non_english_movies(df)
	df = binarize_homepage(df)
	df = add_producers_feature(df)
	df = add_executive_producers_feature(df)
	df = convert_columns_with_list_of_str_to_str(df)
	df = remove_rows_with_non_ascii(df)
	return df
