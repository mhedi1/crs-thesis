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
                    "think": False
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                logger.warning(f"Qwen timeout or error, retrying ({attempt + 1}/2)...")
            else:
                logger.error(f"[QWEN ERROR] {e}")
                raise


def rerank(history: str, candidates: list[dict]) -> dict:
    prompt = build_rerank_prompt(history, candidates)
    try:
        raw_output = call_qwen(prompt)
    except Exception as e:
        logger.warning("[Reranker] Qwen call failed after retries. Using fallback.")
        return candidates[0] if candidates else {"title": "Unknown", "genre": "Unknown", "decade": "Unknown"}

    answer_id = parse_answer_id(raw_output)

    if answer_id is None:
        return candidates[0] if candidates else {"title": "Unknown", "genre": "Unknown", "decade": "Unknown"}

    for candidate in candidates:
        if candidate["id"] == answer_id:
            return candidate

    return candidates[0] if candidates else {"title": "Unknown", "genre": "Unknown", "decade": "Unknown"}