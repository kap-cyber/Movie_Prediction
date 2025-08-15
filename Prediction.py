import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and threshold
loaded_model = joblib.load("xgb_model.pkl")
loaded_thresh = joblib.load("xgb_best_threshold.pkl")

st.title("🎬 Movie Hit or Flop Prediction")

st.write("Enter movie details below to predict if it will be a **Hit** or **Flop**.")

# Input fields
movie_name = st.text_input("Movie Name", "Example Movie")
genre_options = ["Action", "Comedy", "Drama", "Romance", "Thriller"]
genre_mapping = {"Action": 0, "Comedy": 1, "Drama": 2, "Romance": 3, "Thriller": 4}

genre = st.selectbox("Genre", genre_options)
genre_numeric = genre_mapping[genre]

budget = st.number_input("Budget (Cr)", min_value=0.0, step=1.0)
director_success = st.number_input("Director Success Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
actor_success = st.number_input("Lead Actor Success Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
imdb_rating = st.number_input("IMDb Rating", min_value=0.0, max_value=10.0, step=0.1)
trailer_views = st.number_input("Trailer Views", min_value=0, step=1)
trailer_likes = st.number_input("Trailer Likes", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    input_data = np.array([[budget, director_success, actor_success, imdb_rating,
                            trailer_views, trailer_likes, genre_numeric]])
    prob = loaded_model.predict_proba(input_data)[:, 1][0]
    prediction = "Hit" if prob >= loaded_thresh else "Flop"
    st.header(f"Prediction for **{movie_name}**: **{prediction}** (Probability: {prob:.2f})")
