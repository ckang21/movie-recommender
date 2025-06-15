# 🎬 Movie Recommender App

A simple movie recommendation system that uses sentence-transformer embeddings and genre data to suggest similar movies. Built with FastAPI and served through a clean HTML frontend.

## 🚀 Features

- 🔍 Fuzzy-matched movie search
- 🧠 Smart recommendations using sentence embeddings (`all-MiniLM-L6-v2`)
- 🎭 Includes genre data for hybrid content filtering
- 🌐 FastAPI backend with `/recommend` endpoint
- 🖥️ HTML frontend with clean UI and dynamic results
- 📦 Ready for deployment (HuggingFace Spaces, Render, etc.)

## 📸 Demo

<!-- Replace this with your live link after deployment -->
Coming soon!

## 🧠 How it works

1. Loads metadata from the MovieLens 100K dataset
2. Encodes movie titles using [sentence-transformers](https://www.sbert.net/)
3. Combines semantic vectors with binary genre flags
4. Computes cosine similarity between all movies
5. Exposes `/recommend?movie=The Matrix` endpoint to retrieve top 5 similar movies

## 📁 Project Structure

```
movie-recommender/
├── app/
│ ├── api.py # FastAPI app with /recommend endpoint
│ ├── model.py # Recommender logic (embeddings + similarity)
│ └── static/
│ ├── index.html # Frontend UI
│ └── style.css # Frontend styling
├── data/
│ └── ml-100k/ # MovieLens dataset files
├── requirements.txt
└── README.md
```

## 🧪 Example

Request:

```
GET /recommend?movie=Star Wars
```

Response:

```json
{
  "input": "Star Wars",
  "recommendations": [
    "Return of the Jedi (1983)",
    "Empire Strikes Back, The (1980)",
    "Stargate (1994)"
  ]
}
```

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender
cd movie-recommender
pip install -r requirements.txt
python3 -m uvicorn app.api:app --reload
```
Then visit http://127.0.0.1:8000/static/index.html


## 📚 Tech Stack
 - Python

 - FastAPI

 - sentence-transformers

 - RapidFuzz (fuzzy matching)

 - HTML + CSS

## 🧠 Credits
 - MovieLens 100K Dataset
 - Sentence-Transformers