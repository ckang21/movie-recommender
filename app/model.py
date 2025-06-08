import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

# 1️⃣ Load movie metadata
movies_df = pd.read_csv(
    'data/ml-100k/u.item',
    sep='|',
    encoding='latin-1',
    header=None,
    names=[
        'movie_id', 'title', 'release_date', 'video_release_date',
        'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western'
    ]
)

# 2️⃣ TF-IDF on movie titles
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['title'])

# 3️⃣ Genres → binary matrix
genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

genre_matrix = movies_df[genre_cols].values

# 4️⃣ Combine title TF-IDF and genres into one big matrix
import numpy as np
combined_matrix = np.hstack([tfidf_matrix.toarray(), genre_matrix])

# 5️⃣ Compute cosine similarity between all movies
cosine_sim = cosine_similarity(combined_matrix, combined_matrix)

# 6️⃣ Build recommendation function

def recommend(movie_title, top_n=5):
    # Fuzzy match movie title
    choices = movies_df['title'].tolist()
    best_match = process.extractOne(movie_title, choices)

    if best_match is None or best_match[1] < 50:  # confidence threshold
        return ["Movie not found in database."]

    matched_title = best_match[0]
    idx = movies_df[movies_df['title'] == matched_title].index[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]

    return movies_df['title'].iloc[movie_indices].tolist()

