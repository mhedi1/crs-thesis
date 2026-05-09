import logging
from prompts import build_response_prompt
from reranker import call_qwen, USE_FAKE_MODE

logger = logging.getLogger(__name__)


def _fallback_response(movie: dict) -> str:
    title = movie.get("title", "this film")
    genre = movie.get("genre", "")
    decade = movie.get("decade", "")
    parts = []
    if genre and genre != "Unknown":
        parts.append(genre.lower())
    if decade and decade != "Unknown":
        parts.append(f"from the {decade}")
    description = " ".join(parts)
    if description:
        return (f"I would recommend {title}. "
                f"It is a great {description} film that "
                f"matches what you are looking for.")
    else:
        return (f"I would recommend {title}. "
                f"I think it fits well with what you described.")


def generate_response(history: str, selected_movie: dict) -> str:
    if USE_FAKE_MODE:
        return _fallback_response(selected_movie)

    prompt = build_response_prompt(history, selected_movie)

    try:
        response = call_qwen(prompt)
        if not response or not response.strip():
            logger.warning("[Response Generator] Empty output from Qwen.")
            return _fallback_response(selected_movie)
        return response.strip()
    except Exception as e:
        logger.error(f"[Response Generator ERROR] {e}")
        return _fallback_response(selected_movie)