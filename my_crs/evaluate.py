import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\mhfou\Desktop\thesis_crs\baseline_repo\KBRD_project\KBRD")

from kbrd_adapter import get_kbrd_candidates

TEST_DATA_PATH = r"C:\Users\mhfou\Desktop\thesis_crs\baseline_repo\KBRD_project\KBRD\data\redial\test_data.jsonl"


def build_dialogue(sample: dict) -> str:
    """
    Convert ReDial sample to readable dialogue.
    Replaces @movie_id with actual movie name.
    """
    movie_mentions = sample.get("movieMentions", {})
    messages = sample.get("messages", [])

    # Determine which worker is the seeker (initiator)
    initiator = sample.get("initiatorWorkerId", -1)

    turns = []
    for msg in messages:
        role = "User" if msg.get("senderWorkerId") == initiator else "System"
        text = msg.get("text", "").strip()

        # Replace @movie_id with movie name
        for movie_id, movie_name in movie_mentions.items():
            text = text.replace(f"@{movie_id}", movie_name.strip())

        # Clean HTML entities
        text = text.replace("&quot;", '"').replace("&amp;", "&")

        if text:
            turns.append(f"{role}: {text}")

    return "\n".join(turns)


def get_ground_truth(sample: dict) -> list:
    """
    Extract movies that were actually suggested (suggested=1)
    by the respondent (recommender).
    """
    movie_mentions = sample.get("movieMentions", {})
    respondent_questions = sample.get("respondentQuestions", {})

    ground_truth = []
    for movie_id, info in respondent_questions.items():
        if info.get("suggested", 0) == 1:
            movie_name = movie_mentions.get(movie_id, "")
            if movie_name:
                ground_truth.append(movie_name.strip().lower())

    return ground_truth


def normalize_title(title: str) -> str:
    """Clean title for comparison."""
    title = re.sub(r'\(\d{4}\)', '', title)
    title = re.sub(r'[^\w\s]', '', title)
    return title.lower().strip()


def is_hit(candidates: list, ground_truth: list, k: int) -> bool:
    """Check if any ground truth movie is in top-k candidates."""
    top_k = candidates[:k]
    top_k_normalized = [normalize_title(c["title"]) for c in top_k]

    for gt_movie in ground_truth:
        gt_normalized = normalize_title(gt_movie)
        if len(gt_normalized) < 3:
            continue
        for candidate_title in top_k_normalized:
            if gt_normalized in candidate_title or candidate_title in gt_normalized:
                if len(gt_normalized) > 4:
                    return True
    return False


def evaluate(max_samples: int = 200):
    k_values = [1, 10, 50]
    hits = {k: [] for k in k_values}
    total_processed = 0
    total_skipped = 0

    print(f"\n{'='*60}")
    print(f"EVALUATION — ReDial Test Set")
    print(f"Max samples: {max_samples}")
    print(f"{'='*60}\n")

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if total_processed >= max_samples:
                break
            try:
                sample = json.loads(line)

                dialogue = build_dialogue(sample)
                ground_truth = get_ground_truth(sample)

                if not dialogue.strip() or not ground_truth:
                    total_skipped += 1
                    continue

                # Get candidates from KBRD neural model
                candidates = get_kbrd_candidates(dialogue, top_k=50)

                # Check hit at each K
                for k in k_values:
                    hit = is_hit(candidates, ground_truth, k)
                    hits[k].append(hit)

                total_processed += 1

                if total_processed % 20 == 0:
                    r1 = sum(hits[1]) / len(hits[1]) if hits[1] else 0
                    r10 = sum(hits[10]) / len(hits[10]) if hits[10] else 0
                    r50 = sum(hits[50]) / len(hits[50]) if hits[50] else 0
                    print(f"[{total_processed}/{max_samples}] "
                          f"R@1={r1:.4f} R@10={r10:.4f} R@50={r50:.4f}")

            except Exception as e:
                total_skipped += 1
                continue

    # Print final results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Processed: {total_processed}  |  Skipped: {total_skipped}")
    print()
    for k in k_values:
        if hits[k]:
            recall = sum(hits[k]) / len(hits[k])
            print(f"Recall@{k:<3}: {recall:.4f}  "
                  f"({sum(hits[k])}/{len(hits[k])} dialogues)")
    print(f"{'='*60}\n")

    # Save results
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "experiments", "evaluation_results.txt"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("KBRD + Qwen Evaluation Results\n")
        f.write(f"Dataset: ReDial test set\n")
        f.write(f"Samples: {total_processed}\n\n")
        for k in k_values:
            if hits[k]:
                recall = sum(hits[k]) / len(hits[k])
                f.write(f"Recall@{k}: {recall:.4f}\n")

    print(f"Results saved to experiments/evaluation_results.txt")


if __name__ == "__main__":
    evaluate(max_samples=200)