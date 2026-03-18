import streamlit as st, numpy as np, pandas as pd

# Make the web app wide-screen
st.set_page_config(layout="wide")

from IBCF import load_movies, parse_movie_img, myIBCF

st.title("Movie Recommender System")

st.header("Rate your favorite movies!")

# Loads the all the movies and selects 100 of them to be rated by the users
all_movies = load_movies()

displayed_movies = all_movies.head(100)

# Construct the grid of 100 images with the stars for the users to rate
for i in range(10):
    cols = st.columns(10)

    for j in range(10):
        movie = displayed_movies.iloc[10*i + j, :]
        movie_id = movie['movie_id'][1:]
        
        # Parse image and star feedback widgets
        cols[j].image(parse_movie_img(movie_id), caption=movie["title"], use_container_width=True)
        cols[j].feedback("stars", key=movie["movie_id"])

if st.button("Submit"):
   # Convert user ratings to Series
    new_user = pd.Series(st.session_state)

    # Generate recommendations
    recommendations = myIBCF(new_user, top_n=10)

    # Display top 10 recommended movies
    st.header("Top 10 Recommended Movies for You!")
    cols = st.columns(10)
    for i in range(min(10, len(recommendations))): # Ensure no out of bounds errors
        movie_id = recommendations.iloc[i, :]['movie_id']
        movie_title = all_movies[all_movies['movie_id'] == movie_id]['title'].values[0]
        cols[i].image(parse_movie_img(movie_id[1:]), caption=movie_title, use_container_width=True)
