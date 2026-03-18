import pandas as pd, numpy as np, requests, streamlit as st

@st.cache_data
def load_similarity_matrix():
    return pd.read_csv("data/similarity_matrix_top_30.csv", index_col='movie_id')

S = load_similarity_matrix()

@st.cache_data
def load_movies():
    # Define the URL for movie data
    movie_url = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

    # Fetch the data from the URL
    response = requests.get(movie_url)

    # Split the data into lines and then split each line using "::"
    movie_lines = response.text.split('\n')
    movie_data = [line.split("::") for line in movie_lines if line]

    # Create a DataFrame from the movie data
    movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
    movies['movie_id'] = 'm' + movies['movie_id'].astype(str)
    movies = movies[movies['movie_id'].isin(S.columns)]
    
    return movies

all_movies = load_movies()
all_movie_ids = all_movies['movie_id'].sort_values().tolist()
S = S.loc[all_movie_ids, all_movie_ids]

def parse_movie_img(movie_id):
    return f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg?raw=true"

@st.cache_data
def load_top_ten_movies():
    top_ten_movies = pd.read_csv("data/top_10_popular_movies.csv")
    top_ten_movies['movie_img'] = top_ten_movies['movie_img'] + '.jpg?raw=true'
    return top_ten_movies

system1 = load_top_ten_movies()

@st.cache_data
def myIBCF(newuser, top_n=10):
    # Initialize predictions with NaN values for all movies
    predictions = pd.Series(np.nan, index=S.columns)
    rated_movies = newuser.dropna().index # Movies already rated by the user

    # Loop through each movie to predict ratings for unrated ones
    for movie in S.columns:
        if movie not in rated_movies:
            # Get the similar movies
            similar_movies = S.loc[movie, rated_movies].dropna()

            # Calculate numerator and denominator
            numerator = (similar_movies * newuser.loc[similar_movies.index]).sum()
            denominator = similar_movies.sum()

            # Update prediction if the denominator is not zero
            if denominator != 0:
                predictions[movie] = numerator / denominator

    # Convert predictions to a DataFrame for easy sorting and filtering
    pred_df = pd.DataFrame({
        'movie_id': predictions.index,
        'pred_rating': predictions.values
    })

    # Sort by predicted rating and filter out NaN predictions
    top_recommendations = pred_df.sort_values(by='pred_rating', ascending=False).head(top_n)
    valid_predictions = top_recommendations[~top_recommendations['pred_rating'].isna()]

    # If not enough valid predictions, add popular movies
    if len(valid_predictions) < top_n:
        rated_movie_ids = rated_movies
        # Filter popular movies with more than 1000 ratings and sort by average rating
        popular_movies = system1.sort_values(by='avg_rating', ascending=False)
        popular_movies = popular_movies[~popular_movies['movie_id'].isin(rated_movie_ids)]
        popular_movies = popular_movies.head(top_n - len(valid_predictions))

        # Concatenate valid predictions with popular movies to make up top_n recommendations
        top_recommendations = pd.concat([valid_predictions, popular_movies[['movie_id']].head(top_n - len(valid_predictions))])

    return top_recommendations
