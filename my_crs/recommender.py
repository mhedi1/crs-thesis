import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\mhfou\Desktop\thesis_crs\baseline_repo\KBRD_project\KBRD")

from kbrd_adapter import get_kbrd_candidates
from reranker import rerank
from response_generator import generate_response


def get_recommendation(dialogue_history: list) -> dict:
    """
    Main entry point for the CRS pipeline.
    
    Args:
        dialogue_history: list of dicts like:
        [
            {"role": "user", "content": "I want horror"},
            {"role": "system", "content": "Did you see The Shining?"},
            {"role": "user", "content": "Yes, I want something old"}
        ]
    
    Returns:
        {
            "response": "natural language response",
            "movie": {
                "title": "...",
                "genre": "...", 
                "decade": "..."
            },
            "candidates": [top 5 candidates]
        }
    """
    # Convert history list to dialogue string
    turns = []
    for turn in dialogue_history:
        role = "User" if turn["role"] == "user" else "System"
        turns.append(f"{role}: {turn['content']}")
    dialogue_str = "\n".join(turns)
    
    # Stage 1: Get candidates from KBRD
    candidates = get_kbrd_candidates(dialogue_str, top_k=50)
    
    # Stage 2: Rerank with Qwen
    selected_movie = rerank(dialogue_str, candidates)
    
    # Stage 3: Generate response with Qwen
    response = generate_response(dialogue_str, selected_movie)
    
    return {
        "response": response,
        "movie": {
            "title": selected_movie.get("title", "Unknown"),
            "genre": selected_movie.get("genre", "Unknown"),
            "decade": selected_movie.get("decade", "Unknown")
        },
        "candidates": candidates[:5]
    }


if __name__ == "__main__":
    test_history = [
        {"role": "user", 
         "content": "I am looking for a horror film"},
        {"role": "system", 
         "content": "Did you see A Nightmare on Elm Street?"},
        {"role": "user", 
         "content": "Yes I have seen it. I want something old."}
    ]
    
    result = get_recommendation(test_history)
    print("Movie:", result["movie"]["title"])
    print("Genre:", result["movie"]["genre"])
    print("Decade:", result["movie"]["decade"])
    print("Response:", result["response"])
