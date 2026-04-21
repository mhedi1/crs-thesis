import re
from typing import List, Dict, Any

MOVIE_DB = [
    {"id": 1, "title": "It (2017)", "genre": "Horror", "decade": "2010s", "keywords": ["clown", "scary", "modern"]},
    {"id": 2, "title": "The Shining (1980)", "genre": "Horror", "decade": "1980s", "keywords": ["old", "classic", "scary", "hotel"]},
    {"id": 3, "title": "Carrie (1976)", "genre": "Horror", "decade": "1970s", "keywords": ["old", "classic", "scary", "school"]},
    {"id": 4, "title": "Halloween (1978)", "genre": "Horror", "decade": "1970s", "keywords": ["old", "classic", "scary", "slasher", "michael", "myers"]},
    {"id": 5, "title": "Superbad (2007)", "genre": "Comedy", "decade": "2000s", "keywords": ["funny", "teen", "school"]},
    {"id": 6, "title": "The Godfather (1972)", "genre": "Crime", "decade": "1970s", "keywords": ["old", "classic", "mafia", "masterpiece"]},
    {"id": 7, "title": "Toy Story (1995)", "genre": "Animation", "decade": "1990s", "keywords": ["family", "classic", "kids", "funny"]},
    {"id": 8, "title": "Avengers: Endgame (2019)", "genre": "Action", "decade": "2010s", "keywords": ["superhero", "marvel", "modern", "action"]},
    {"id": 9, "title": "Back to the Future (1985)", "genre": "Sci-Fi", "decade": "1980s", "keywords": ["classic", "funny", "time", "80s"]},
    {"id": 10, "title": "La La Land (2016)", "genre": "Romance", "decade": "2010s", "keywords": ["music", "romantic", "modern"]},
]


def extract_preferences(dialogue: str) -> List[str]:
    """
    Extract simple preference clues from dialogue.
    """
    target_keywords = {
        "horror", "comedy", "crime", "animation", "action", "sci", "sci-fi", "romance",
        "old", "classic", "modern", "scary", "funny", "superhero", "family",
        "70s", "80s", "90s", "1970s", "1980s", "1990s", "2000s", "2010s"
    }

    words = re.findall(r"\b[\w-]+\b", dialogue.lower())
    extracted = []

    for word in words:
        if word in target_keywords and word not in extracted:
            extracted.append(word)

    # normalize short decade forms
    normalized = []
    for item in extracted:
        if item == "70s":
            normalized.append("1970s")
        elif item == "80s":
            normalized.append("1980s")
        elif item == "90s":
            normalized.append("1990s")
        elif item == "sci":
            normalized.append("sci-fi")
        else:
            normalized.append(item)

    return normalized


def score_movie(movie: Dict[str, Any], preferences: List[str]) -> int:
    """
    Score a movie against extracted preferences.
    """
    score = 0
    movie_genre = movie.get("genre", "").lower()
    movie_decade = movie.get("decade", "").lower()
    movie_keywords = [kw.lower() for kw in movie.get("keywords", [])]

    for pref in preferences:
        if pref == movie_genre:
            score += 2
        if pref == movie_decade:
            score += 2
        if pref in movie_keywords:
            score += 1

    return score


def generate_candidates(dialogue: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Generate top-k candidate movies from the local database.
    """
    preferences = extract_preferences(dialogue)

    scored_movies = []
    for movie in MOVIE_DB:
        score = score_movie(movie, preferences)
        scored_movies.append({"movie": movie, "score": score})

    scored_movies.sort(key=lambda x: x["score"], reverse=True)

    positive = [m for m in scored_movies if m["score"] > 0]

    if positive:
        return [m["movie"] for m in positive[:top_k]]

    return [m["movie"] for m in scored_movies[:top_k]]