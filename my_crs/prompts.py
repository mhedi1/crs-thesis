def build_rerank_prompt(history: str, candidates: list[dict]) -> str:
    """
    Build the prompt for Qwen reranking.
    Each candidate should have:
    - id
    - title
    - genre
    - decade
    """
    candidate_lines = []
    for c in candidates:
        candidate_lines.append(
            f"{c['id']}. {c['title']} | genres: {c.get('genre', 'unknown')} | decade: {c.get('decade', 'unknown')}"
        )

    candidate_block = "\n".join(candidate_lines)

    prompt = f"""You are a movie recommendation reranker.

Select the single best movie from the candidate list based on the dialogue.
Follow explicit user constraints strictly.
Prefer candidates that match genre, year/era, and semantic clues from the conversation.

Return only:
ANSWER: <candidate_id>

Dialogue:
{history}

Candidates:
{candidate_block}
"""
    return prompt


def build_response_prompt(history: str, selected_movie: dict) -> str:
    """
    Build the prompt for natural response generation.
    """
    prompt = f"""You are a helpful and natural movie recommender.

Write one conversational response using the selected recommendation.

The reply must:
- mention the movie naturally,
- justify it with the user's preferences,
- sound concise and friendly,
- avoid mentioning unselected candidates.

Dialogue:
{history}

Selected recommendation:
{selected_movie['title']} | genres: {selected_movie.get('genre', 'unknown')} | decade: {selected_movie.get('decade', 'unknown')}
"""
    return prompt