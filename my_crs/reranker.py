import requests
from prompts import build_rerank_prompt
from utils import parse_answer_id

LLM_URL = "http://sinbad2ia.ujaen.es:8050/api/chat"
MODEL_NAME = "qwen3.5:35b"
USE_FAKE_MODE = True


def call_qwen(prompt: str) -> str:
    if USE_FAKE_MODE:
        # Temporary fake output for local testing
        prompt_lower = prompt.lower()

        if "superhero" in prompt_lower or "marvel" in prompt_lower or "modern" in prompt_lower:
            return "ANSWER: 8"

        if "funny" in prompt_lower or "family" in prompt_lower or "animation" in prompt_lower:
            return "ANSWER: 7"

        if "horror" in prompt_lower or "old" in prompt_lower or "classic" in prompt_lower:
            return "ANSWER: 4"

        return "ANSWER: 1"

    response = requests.post(
        LLM_URL,
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def rerank(history: str, candidates: list[dict]) -> dict:
    prompt = build_rerank_prompt(history, candidates)
    raw_output = call_qwen(prompt)

    print("\n[DEBUG] Raw reranker output:")
    print(raw_output)

    answer_id = parse_answer_id(raw_output)

    if answer_id is None:
        return candidates[0]

    for candidate in candidates:
        if candidate["id"] == answer_id:
            return candidate

    return candidates[0]