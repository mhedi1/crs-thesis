import requests
import logging
from prompts import build_rerank_prompt
from utils import parse_answer_id

logger = logging.getLogger(__name__)

LLM_URL = "http://sinbad2ia.ujaen.es:8050/api/chat"
MODEL_NAME = "qwen3.5:35b"
USE_FAKE_MODE = False


def call_qwen(messages) -> str:
    if USE_FAKE_MODE:
        if isinstance(messages, list):
            prompt = messages[-1]["content"]
        else:
            prompt = messages
        prompt_lower = prompt.lower()

        if "superhero" in prompt_lower or "marvel" in prompt_lower or "modern" in prompt_lower:
            return "ANSWER: 8"

        if "funny" in prompt_lower or "family" in prompt_lower or "animation" in prompt_lower:
            return "ANSWER: 7"

        if "horror" in prompt_lower or "old" in prompt_lower or "classic" in prompt_lower:
            return "ANSWER: 4"

        return "ANSWER: 1"

    if isinstance(messages, str):
        payload_messages = [{"role": "user", "content": messages}]
    else:
        payload_messages = messages

    for attempt in range(3):
        try:
            response = requests.post(
                LLM_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": payload_messages,
                    "stream": False,
                    "think": False,
                    "temperature": 0,
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                logger.warning(f"Qwen timeout or error, retrying ({attempt + 1}/3)...")
            else:
                logger.error(f"[QWEN ERROR] {e}")
                raise


def rerank(history: str, candidates: list[dict], era_hints: list = None, serialization_format: int = 3) -> tuple[dict, bool]:
    """Returns (selected_movie, is_fallback)"""
    prompt = build_rerank_prompt(history, candidates, era_hints=era_hints, serialization_format=serialization_format)
    
    try:
        raw_output = call_qwen(prompt)
    except Exception as e:
        logger.warning("[Reranker] Qwen call failed after retries. Using fallback.")
        return (candidates[0] if candidates else {"title": "Unknown", "genre": "Unknown", "decade": "Unknown"}, True)

    answer_idx = parse_answer_id(raw_output)

    if answer_idx is None:
        logger.warning("[Reranker] Could not parse answer. Using fallback.")
        return (candidates[0] if candidates else {"title": "Unknown", "genre": "Unknown", "decade": "Unknown"}, True)

    if answer_idx < 1 or answer_idx > len(candidates):
        logger.warning(f"[Reranker] Answer index {answer_idx} out of range. Using fallback.")
        return (candidates[0] if candidates else {"title": "Unknown", "genre": "Unknown", "decade": "Unknown"}, True)

    selected = candidates[answer_idx - 1]
    logger.info(f"[Reranker] Qwen selected position {answer_idx}: {selected['title']}")
    return (selected, False)