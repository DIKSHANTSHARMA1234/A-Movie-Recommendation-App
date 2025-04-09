# 1. Import all the required Python libraries for the app
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 2. Set up the Flask web application
app = Flask(__name__)

# 3. Load and clean the movie dataset using pandas
movies = pd.read_csv('cleaned_movies.csv')
movies['title'] = movies['title'].fillna('').astype(str)
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('[]')
movies['keywords'] = movies['keywords'].fillna('[]')
movies['title_lower'] = movies['title'].str.lower().str.strip()

# 4. Create a function to combine movie overview, genres, and keywords
def combine_features(row):
    return row['overview'] + ' ' + row['genres'] + ' ' + row['keywords']

# 5. Generate TF-IDF vectors and compute cosine similarity
movies['combined_features'] = movies.apply(combine_features, axis=1)
tfidf = TfidfVectorizer(stop_words='english')
vector_matrix = tfidf.fit_transform(movies['combined_features'])
similarity = cosine_similarity(vector_matrix)

# 6. Save processed data and models to files
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)
movies.to_csv('cleaned_movies.csv', index=False)

# 7. Write a function to fetch movie posters using the TMDB API
API_KEY = 'b9d042db15dafa83265ce62283d0db18'
def fetch_poster_image(title):
    try:
        url = f'https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={title}'
        response = requests.get(url, timeout=5)
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f'https://image.tmdb.org/t/p/w500{poster_path}'
        return "/static/no_poster.jpg"
    except Exception as e:
        print("TMDB API Error:", e)
        return "/static/no_poster.jpg"

# 8. Create a function to return poster links for recommended movies
def display_posters(movie_titles):
    posters = []
    for title in movie_titles:
        posters.append(fetch_poster_image(title))
    return posters

# 9. (Optional) Return movie details in a dictionary format
def display_movie_details(movie):
    return {
        'title': movie['title'],
        'overview': movie['overview'],
        'genres': movie['genres'],
        'keywords': movie['keywords']
    }

# 10. Recommend the top 5 most similar movies to a given title
def recommend_movies(title):
    title = title.lower().strip()
    if title not in movies['title_lower'].values:
        return []
    idx = movies[movies['title_lower'] == title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]['title'].values.tolist()

# 11. Pair recommended movie titles with their posters
def display_recommendations(title):
    recommended_titles = recommend_movies(title)
    posters = display_posters(recommended_titles) if recommended_titles else []
    return list(zip(recommended_titles, posters))

# 12. Add a basic command-line test to print recommendations
def main():
    title = input("Enter movie title: ")
    recs = display_recommendations(title)
    if recs:
        for r in recs:
            print("Movie:", r[0])
            print("Poster:", r[1])
            print()
    else:
        print("No matches found.")

# 13. Handle the form input in Flask and show recommendations on the page
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    if request.method == "POST":
        title = request.form["movie"]
        print("User entered:", title)
        try:
            recommendations = display_recommendations(title)
        except Exception as e:
            print("Recommendation error:", e)
            recommendations = None
    return render_template("index.html", recommendations=recommendations)

# 14. Run the Flask app in debug mode
if __name__ == "__main__":
    app.run(debug=True)
