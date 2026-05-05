import os
import pickle
import re
import difflib
import spacy
from typing import List, Dict, Any

GENRE_KEYWORDS = {
    "Horror": ["horror", "scary", "ghost", "zombie", "vampire", "slasher", "haunting", "exorcist", "amityville"],
    "Comedy": ["comedy", "funny", "humor", "laugh"],
    "Animation": ["animated", "animation", "pixar", "disney"],
    "Action": ["action", "superhero", "marvel", "avengers", "batman", "spider"],
    "Drama": ["drama"],
    "Sci-Fi": ["science_fiction", "sci-fi", "space", "alien", "robot"],
    "Crime": ["crime", "gangster", "mafia", "thriller"],
    "Romance": ["romance", "romantic", "love"]
}

KNOWN_MOVIE_GENRES = {
    "halloween": "Horror",
    "carrie": "Horror",
    "psycho": "Horror",
    "jaws": "Horror",
    "the thing": "Horror",
    "alien": "Sci-Fi",
    "blade runner": "Sci-Fi",
    "star wars": "Sci-Fi",
    "the godfather": "Crime",
    "goodfellas": "Crime",
    "scarface": "Crime",
    "toy story": "Animation",
    "finding nemo": "Animation",
    "the lion king": "Animation",
    "titanic": "Romance",
    "pretty woman": "Romance",
    "die hard": "Action",
    "terminator": "Action",
    "rocky": "Drama",
    "forrest gump": "Drama"
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KBRD_REPO_PATH = os.path.normpath(
    os.path.join(CURRENT_DIR, "..", "baseline_repo", "KBRD_project", "KBRD")
)

_data_loaded = False
_has_error = False
_id2entity = None
_movie_ids = None
_entity2id = None
_movie_title_to_id = {}


def get_fallback_candidates(top_k: int) -> List[Dict[str, Any]]:
    return [
        {"id": 101, "title": "The Matrix (1999)", "genre": "Sci-Fi", "decade": "1990s"},
        {"id": 102, "title": "The Exorcist (1973)", "genre": "Horror", "decade": "1970s"},
        {"id": 103, "title": "Shrek (2001)", "genre": "Animation", "decade": "2000s"},
        {"id": 104, "title": "Die Hard (1988)", "genre": "Action", "decade": "1980s"},
        {"id": 105, "title": "The Hangover (2009)", "genre": "Comedy", "decade": "2000s"},
    ][:top_k]


def _load_kbrd_resources() -> None:
    global _data_loaded, _has_error, _id2entity, _movie_ids, _entity2id, _movie_title_to_id

    if _data_loaded or _has_error:
        return

    print("[KBRD Adapter] Loading processed KBRD resources...")

    data_dir = os.path.join(KBRD_REPO_PATH, "data", "redial")
    entity2id_path = os.path.join(data_dir, "entity2entityId.pkl")
    movie_ids_path = os.path.join(data_dir, "movie_ids.pkl")

    try:
        with open(entity2id_path, "rb") as f:
            _entity2id = pickle.load(f)

        with open(movie_ids_path, "rb") as f:
            _movie_ids = pickle.load(f)

        _id2entity = {v: k for k, v in _entity2id.items()}

        # Build fast lowercased lookup for movies to enable exact and fuzzy matching
        for mid in _movie_ids:
            uri = _id2entity.get(mid)
            if uri:
                clean_t = _clean_title(uri).lower()
                if _is_valid_movie_title(clean_t):
                    clean_lookup = re.sub(r"[^\w\s]", "", clean_t).strip()
                    _movie_title_to_id[clean_lookup] = mid

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


def _is_valid_movie_title(title: str) -> bool:
    t_lower = title.lower()

    if t_lower.startswith("list of ") or t_lower.startswith("category: "):
        return False

    if "(film series)" in t_lower or "(tv series)" in t_lower:
        return False

    if "(franchise)" in t_lower:
        return False

    blocklist = ["carriers", "films", "movies", "cinema"]
    if t_lower in blocklist:
        return False

    return True


def _extract_year(text: str) -> str:
    """
    Extracts the release year from a DBpedia URI or title string based on a strict priority order
    to prevent false positives from generic numbers in the URL.
    Valid years are between 1880 and 2030.
    """
    text_str = str(text)
    
    text_lower = text_str.lower()
    
    # 1. Skip if URI contains novel, book, or character
    if "novel" in text_lower or "book" in text_lower or "character" in text_lower:
        return ""
        
    year = ""
    
    # 2. If it contains "_film", strictly require the year to be adjacent to it
    if "_film" in text_lower:
        match_1 = re.search(r"\((18[8-9]\d|19\d{2}|20[0-2]\d|2030)_film\)", text_str, re.IGNORECASE)
        if match_1:
            year = match_1.group(1)
    else:
        # 3. Existing priority order for all other cases
        match_2 = re.search(r"\((18[8-9]\d|19\d{2}|20[0-2]\d|2030)\)", text_str)
        if match_2:
            year = match_2.group(1)
        else:
            match_3 = re.search(r"_(18[8-9]\d|19\d{2}|20[0-2]\d|2030)_", text_str)
            if match_3:
                year = match_3.group(1)

    # 4. Minimum threshold check: if year is before 1900, discard it
    if year and int(year) >= 1900:
        return year
        
    return ""


def _year_to_decade(year: str) -> str:
    if not year:
        return "Unknown"
    decade = int(year[:3]) * 10
    return f"{decade}s"


def _infer_genre(uri: str, title: str) -> str:
    """
    Infers the genre of a movie.
    First checks a hardcoded dictionary for exact title matches (ignoring years).
    Then falls back to keyword matching in the DBpedia URI and clean title.
    Returns the first matching genre, or 'Unknown' if nothing matches.
    """
    # 1. First layer: Check known movie genres by cleaned title
    # Clean the title: lowercase and strip out any trailing (YYYY) years
    clean_title = title.lower()
    clean_title = re.sub(r"\s*\(\d{4}\)", "", clean_title).strip()
    
    if clean_title in KNOWN_MOVIE_GENRES:
        return KNOWN_MOVIE_GENRES[clean_title]

    # 2. Second layer: Keyword matching
    combined_str = (str(uri) + " " + str(title)).lower()
    
    for genre, keywords in GENRE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined_str:
                return genre
                
    return "Unknown"


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


def _score_movie(title: str, prefs: Dict[str, Any], year: str) -> int:
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
        if not _is_valid_movie_title(title):
            continue

        year = _extract_year(entity_uri)
        score = _score_movie(title, prefs, year)

        if score > 0:
            scored.append(
                {
                    "score": score,
                    "id": int(movie_entity_id),
                    "title": title,
                    "genre": _infer_genre(entity_uri, title),
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


def _get_ngrams(words: List[str], n: int) -> List[str]:
    """Helper function to generate n-grams from a list of words."""
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


_nlp = None

def _get_spacy_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def prepare_input(dialogue: str) -> List[int]:
    """
    Hybrid entity extraction & linking module (spaCy + N-grams).
    1. Extracts spaCy named entities and noun chunks.
    2. Generates 1, 2, and 3-grams from dialogue.
    3. Matches against known movies and generic DBpedia URIs.
    4. Uses exact, case-insensitive, and fuzzy matching.
    5. Maps genres dynamically.
    """
    print("[KBRD Adapter] -> STAGE 2: Preparing input from dialogue...")
    _load_kbrd_resources()

    if _has_error or not _entity2id:
        print("[KBRD Adapter WARNING] Skipping input preparation due to prior errors.")
        return []

    # Step A: Preprocessing
    clean_dialogue = re.sub(r"[^\w\s]", "", dialogue.lower()).strip()
    words = clean_dialogue.split()

    seed_set = set()
    detected_phrases = []

    # Step B: spaCy Extraction
    nlp = _get_spacy_nlp()
    doc = nlp(dialogue)
    spacy_phrases = [ent.text.lower() for ent in doc.ents]
    spacy_phrases.extend([chunk.text.lower() for chunk in doc.noun_chunks])
    
    # Clean spaCy phrases
    spacy_phrases = [re.sub(r"[^\w\s]", "", p).strip() for p in spacy_phrases]
    spacy_phrases = [p for p in spacy_phrases if p]

    # Step C: N-Gram Fallback
    all_ngrams = []
    for n in [3, 2, 1]:
        all_ngrams.extend(_get_ngrams(words, n))
        
    # Combine and remove duplicates, preserving order (longest/spaCy first)
    candidate_phrases = []
    for p in spacy_phrases + all_ngrams:
        if p not in candidate_phrases:
            candidate_phrases.append(p)
            
    # Sort candidate phrases by length descending to prioritize longer phrases
    candidate_phrases.sort(key=len, reverse=True)

    # Step D: Matching pipeline
    unmatched_phrases = []
    
    for phrase in candidate_phrases:
        matched = False
        
        # Stage 1: Exact Match
        if phrase in _movie_title_to_id:
            mid = _movie_title_to_id[phrase]
            if mid not in seed_set:
                seed_set.add(mid)
                detected_phrases.append(f"'{phrase}' (Exact Movie Match)")
            matched = True

        # Stage 2: DBpedia URI Match
        if not matched:
            capitalized = "_".join([w.capitalize() for w in phrase.split()])
            potential_uri = f"<http://dbpedia.org/resource/{capitalized}>"
            if potential_uri in _entity2id:
                eid = _entity2id[potential_uri]
                if eid not in seed_set:
                    seed_set.add(eid)
                    detected_phrases.append(f"'{phrase}' (DBpedia URI Match)")
                matched = True
                
        if not matched:
            unmatched_phrases.append(phrase)

    # Stage 3: Fuzzy Matching on unmatched phrases (length > 4)
    long_unmatched = [p for p in unmatched_phrases if len(p) > 4]
    movie_titles = list(_movie_title_to_id.keys())
    
    for phrase in long_unmatched:
        # Prevent redundant fuzzy matching if an exact match already snagged this concept
        if any(phrase in dp for dp in detected_phrases):
            continue
            
        matches = difflib.get_close_matches(phrase, movie_titles, n=3, cutoff=0.8)
        if matches:
            matched_title = matches[0]
            mid = _movie_title_to_id[matched_title]
            if mid not in seed_set:
                seed_set.add(mid)
                detected_phrases.append(f"'{phrase}' -> '{matched_title}' (Fuzzy Movie Match)")

    # Step E: Genre Detection
    genre_map = {
        "horror": "Horror_film",
        "comedy": "Comedy_film",
        "action": "Action_film",
        "animation": "Animated_film",
        "sci fi": "Science_fiction_film",
        "scifi": "Science_fiction_film",
        "thriller": "Thriller_(genre)",
        "romance": "Romance_film",
        "documentary": "Documentary_film",
        "family": "Children's_film",
    }

    # Match individual words against genre map
    for word in words:
        if word in genre_map:
            genre_uri = f"<http://dbpedia.org/resource/{genre_map[word]}>"
            if genre_uri in _entity2id:
                eid = _entity2id[genre_uri]
                if eid not in seed_set:
                    seed_set.add(eid)
                    detected_phrases.append(f"'{word}' (Genre Mapping)")

    # Step F & G: Deduplication and Logging
    seed_list = list(seed_set)
    if not seed_list:
        print("[KBRD Adapter WARNING] No matching entities or movies found in dialogue.")
    else:
        print(f"[KBRD Adapter] Detected Entities:")
        for dp in detected_phrases:
            print(f"  - {dp}")
        print(f"[KBRD Adapter] Found {len(seed_list)} DBpedia entities linked to dialogue.")

    return seed_list