from prompts import build_response_prompt
from reranker import call_qwen, USE_FAKE_MODE


def article_for(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def generate_response(history: str, selected_movie: dict) -> str:
    title = selected_movie["title"]
    genre = selected_movie.get("genre", "Unknown genre").lower()
    decade = selected_movie.get("decade", "Unknown decade")

    article = article_for(genre)

    return (
    f"I would recommend {title}. It is {article} {genre} movie from the {decade}. "
    f"Based on your preferences, it seems like a good match for what you're looking for."
)