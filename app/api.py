from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from app.model import recommend

app = FastAPI()

# Optional: allow calls from anywhere (helpful if you add frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend")
def get_recommendations(movie: str = Query(..., description="Movie title to recommend similar movies for")):
    recommendations = recommend(movie)
    return {
        "input": movie,
        "recommendations": recommendations
    }
