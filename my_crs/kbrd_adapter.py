import os
import pickle
import re
from typing import List, Dict, Any


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KBRD_REPO_PATH = os.path.normpath(
    os.path.join(CURRENT_DIR, "..", "baseline_repo", "KBRD_project", "KBRD")
)

_data_loaded = False
_has_error = False
_id2entity = None
_movie_ids = None


def get_fallback_candidates(top_k: int) -> List[Dict[str, Any]]:
    return [
        {"id": 101, "title": "The Matrix (1999)", "genre": "Sci-Fi", "decade": "1990s"},
        {"id": 102, "title": "The Exorcist (1973)", "genre": "Horror", "decade": "1970s"},
        {"id": 103, "title": "Shrek (2001)", "genre": "Animation", "decade": "2000s"},
        {"id": 104, "title": "Die Hard (1988)", "genre": "Action", "decade": "1980s"},
        {"id": 105, "title": "The Hangover (2009)", "genre": "Comedy", "decade": "2000s"},
    ][:top_k]


def _load_kbrd_resources() -> None:
    global _data_loaded, _has_error, _id2entity, _movie_ids

    if _data_loaded or _has_error:
        return

    print("[KBRD Adapter] Loading processed KBRD resources...")

    data_dir = os.path.join(KBRD_REPO_PATH, "data", "redial")
    entity2id_path = os.path.join(data_dir, "entity2entityId.pkl")
    movie_ids_path = os.path.join(data_dir, "movie_ids.pkl")

    try:
        with open(entity2id_path, "rb") as f:
            entity2id = pickle.load(f)

        with open(movie_ids_path, "rb") as f:
            _movie_ids = pickle.load(f)

        _id2entity = {v: k for k, v in entity2id.items()}
        _data_loaded = True

        print(f"[KBRD Adapter] Loaded {len(_id2entity)} entities and {len(_movie_ids)} movie ids.")

    except Exception as e:
        print(f"[KBRD Adapter ERROR] Could not load KBRD resources: {e}")
        _has_error = True


def _clean_title(entity_uri: str) -> str:
    match = re.search(r"resource/(.+)>", str(entity_uri))
    title = match.group(1) if match else str(entity_uri)

    title = title.replace("_", " ")
    title = re.sub(r"\s*\(film\)", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(\d{4} film\)", "", title, flags=re.IGNORECASE)
    title = title.strip()

    return title


def _extract_year(title: str) -> str:
    match = re.search(r"(19\d{2}|20\d{2})", title)
    return match.group(1) if match else ""


def _year_to_decade(year: str) -> str:
    if not year:
        return "Unknown"
    decade = int(year[:3]) * 10
    return f"{decade}s"


def _extract_preferences(dialogue: str) -> Dict[str, Any]:
    text = dialogue.lower()

    prefs = {
        "horror": any(w in text for w in ["horror", "scary", "slasher"]),
        "comedy": any(w in text for w in ["funny", "comedy", "laugh"]),
        "animation": any(w in text for w in ["animation", "animated", "family", "kids"]),
        "superhero": any(w in text for w in ["superhero", "marvel", "avengers"]),
        "old": any(w in text for w in ["old", "classic", "older"]),
        "modern": any(w in text for w in ["modern", "recent", "new"]),
    }

    decade_match = re.search(r"(1970s|1980s|1990s|2000s|2010s|70s|80s|90s)", text)
    prefs["decade"] = decade_match.group(1) if decade_match else None

    return prefs


def _score_movie(title: str, prefs: Dict[str, Any]) -> int:
    t = title.lower()
    score = 0

    if prefs["horror"] and any(w in t for w in ["horror", "shining", "halloween", "exorcist", "carrie"]):
        score += 5

    if prefs["comedy"] and any(w in t for w in ["comedy", "funny", "hangover", "superbad"]):
        score += 5

    if prefs["animation"] and any(w in t for w in ["toy story", "shrek", "animation"]):
        score += 5

    if prefs["superhero"] and any(w in t for w in ["avengers", "spider", "batman", "superman", "marvel"]):
        score += 5

    year = _extract_year(title)
    if year:
        y = int(year)
        if prefs["old"] and y < 1990:
            score += 3
        if prefs["modern"] and y >= 2000:
            score += 3

    return score


def get_kbrd_candidates(dialogue: str, top_k: int = 5) -> List[Dict[str, Any]]:
    print(f"\n{'=' * 50}")
    print("[KBRD Adapter] Starting Resource-Based Candidate Generation")
    print(f"{'=' * 50}")

    _load_kbrd_resources()

    if _has_error or not _data_loaded:
        print("[KBRD Adapter] Falling back because resources are unavailable.")
        return get_fallback_candidates(top_k)

    prefs = _extract_preferences(dialogue)

    scored = []
    for movie_entity_id in _movie_ids:
        entity_uri = _id2entity.get(movie_entity_id)
        if not entity_uri:
            continue

        title = _clean_title(entity_uri)
        score = _score_movie(title, prefs)

        if score > 0:
            year = _extract_year(title)
            scored.append(
                {
                    "score": score,
                    "id": int(movie_entity_id),
                    "title": title,
                    "genre": "Unknown",
                    "decade": _year_to_decade(year),
                    "source": "KBRD_PROCESSED_DATA",
                }
            )

    scored.sort(key=lambda x: x["score"], reverse=True)

    candidates = [
        {
            "id": item["id"],
            "title": item["title"],
            "genre": item["genre"],
            "decade": item["decade"],
            "source": item["source"],
        }
        for item in scored[:top_k]
    ]

    if not candidates:
        print("[KBRD Adapter] No matching candidates found. Using fallback.")
        return get_fallback_candidates(top_k)

    print(f"[KBRD Adapter] Returning {len(candidates)} candidates from processed KBRD resources.")
    return candidates