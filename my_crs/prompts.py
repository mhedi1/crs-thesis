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
    for i, c in enumerate(candidates):
        candidate_lines.append(
            f"{i+1}. {c['title']} | "
            f"genres: {c.get('genre', 'Unknown')} | "
            f"decade: {c.get('decade', 'Unknown')}"
        )

    candidate_block = "\n".join(candidate_lines)

    messages = [
        {"role": "system", "content": "You are a movie recommendation reranker."},
        {"role": "user", "content": f"""Select the single best movie from the candidate list based on the dialogue.
Follow explicit user constraints strictly.
Prefer candidates that match genre, year/era, and semantic clues from the conversation.
Pay special attention to exclusions — if the user has already seen a movie, mentioned disliking it, or asked for something different, do NOT select that movie even if it appears in the candidate list.

Return only:
ANSWER: <number>
where <number> is the position in the list (1 to 50)

Dialogue:
{history}

Candidates:
{candidate_block}"""}
    ]
    return messages


def build_response_prompt(history: str, selected_movie: dict, reason_hints: str = None) -> str:
    """
    Build the prompt for natural response generation.
    """
    if reason_hints is None:
        lines = history.split('\n')
        last_user_msg = ""
        for line in reversed(lines):
            if line.startswith("User:"):
                last_user_msg = line
                break
        
        if last_user_msg.startswith("User:"):
            last_user_msg = last_user_msg[5:].strip()
            
        reason_hints = last_user_msg[:50]
        if not reason_hints:
            reason_hints = "matches user preference from conversation"

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
{selected_movie['title']} | genres: {selected_movie.get('genre', 'unknown')} | decade: {selected_movie.get('decade', 'unknown')} | reason hints: {reason_hints}
"""
    return prompt