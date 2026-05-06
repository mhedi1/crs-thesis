import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\mhfou\Desktop\thesis_crs\baseline_repo\KBRD_project\KBRD")

from kbrd_adapter import get_kbrd_candidates

TEST_DATA_PATH = r"C:\Users\mhfou\Desktop\thesis_crs\baseline_repo\KBRD_project\KBRD\data\redial\test_data.jsonl"


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


def build_dialogue_up_to(sample: dict, turn_index: int) -> str:
    """
    Builds dialogue text using only messages up to and including the given turn index.
    Replaces @movie_id with movie name.
    Cleans HTML entities.
    """
    movie_mentions = sample.get("movieMentions", {})
    messages = sample.get("messages", [])
    initiator = sample.get("initiatorWorkerId", -1)

    turns = []
    for i, msg in enumerate(messages):
        if i > turn_index:
            break
            
        role = "User" if msg.get("senderWorkerId") == initiator else "System"
        text = msg.get("text", "").strip()

        for movie_id, movie_name in movie_mentions.items():
            text = text.replace(f"@{movie_id}", movie_name.strip())

        text = text.replace("&quot;", '"').replace("&amp;", "&")

        if text:
            turns.append(f"{role}: {text}")

    return "\n".join(turns)


def get_recommended_movies_at_turn(sample: dict, turn_index: int) -> list:
    """
    Looks at the message at turn_index.
    Finds all @movie_id references in the text.
    Returns only those movies where "suggested" equals 1 in respondentQuestions.
    """
    messages = sample.get("messages", [])
    if turn_index >= len(messages):
        return []
        
    msg = messages[turn_index]
    text = msg.get("text", "")
    movie_mentions = sample.get("movieMentions", {})
    respondent_questions = sample.get("respondentQuestions", {})

    movie_ids_in_text = re.findall(r'@(\d+)', text)
    
    recommended_movies = []
    for movie_id in movie_ids_in_text:
        info = respondent_questions.get(movie_id, {})
        if info.get("suggested", 0) == 1:
            movie_name = movie_mentions.get(movie_id, "")
            if movie_name:
                recommended_movies.append(movie_name.strip().lower())
                
    return recommended_movies


def evaluate(max_samples: int = 200):
    k_values = [1, 10, 50]
    hits = {k: [] for k in k_values}
    
    total_conversations_processed = 0
    total_evaluation_instances = 0

    print(f"\n{'='*60}")
    print(f"EVALUATION — ReDial Test Set (Turn-by-Turn)")
    print(f"Max conversations to process: {max_samples}")
    print(f"{'='*60}\n")

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if total_conversations_processed >= max_samples:
                break
                
            try:
                sample = json.loads(line)
                messages = sample.get("messages", [])
                respondent = sample.get("respondentWorkerId", -1)
                
                conversation_has_instances = False
                
                for turn_index, msg in enumerate(messages):
                    sender = msg.get("senderWorkerId", -1)
                    
                    if sender == respondent:
                        # System turn
                        recommended_movies = get_recommended_movies_at_turn(sample, turn_index)
                        
                        if recommended_movies:
                            # Evaluation instance
                            dialogue_up_to = build_dialogue_up_to(sample, turn_index)
                            
                            candidates = get_kbrd_candidates(dialogue_up_to, top_k=50)
                            
                            for k in k_values:
                                hit = is_hit(candidates, recommended_movies, k)
                                hits[k].append(hit)
                                
                            total_evaluation_instances += 1
                            conversation_has_instances = True
                            
                            if total_evaluation_instances % 20 == 0:
                                r1 = sum(hits[1]) / len(hits[1]) if hits[1] else 0
                                r10 = sum(hits[10]) / len(hits[10]) if hits[10] else 0
                                r50 = sum(hits[50]) / len(hits[50]) if hits[50] else 0
                                print(f"[{total_evaluation_instances} instances] "
                                      f"R@1={r1:.4f} R@10={r10:.4f} R@50={r50:.4f}")
                                      
                if conversation_has_instances:
                    total_conversations_processed += 1
                    
            except Exception as e:
                # print(f"Error: {e}")
                continue

    # Final Results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total conversations processed: {total_conversations_processed}")
    print(f"Total evaluation instances: {total_evaluation_instances}")
    print()
    
    final_recalls = {}
    for k in k_values:
        if hits[k]:
            recall = sum(hits[k]) / len(hits[k])
            final_recalls[k] = recall
            print(f"Recall@{k:<3}: {recall:.4f}  "
                  f"({sum(hits[k])}/{len(hits[k])} instances)")
                  
    print(f"{'='*60}\n")
    
    # Save results
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "experiments", "evaluation_turn_by_turn.txt"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("KBRD + Qwen Turn-by-Turn Evaluation Results\n")
        f.write(f"Dataset: ReDial test set\n")
        f.write(f"Conversations: {total_conversations_processed}\n")
        f.write(f"Instances: {total_evaluation_instances}\n\n")
        for k in k_values:
            if hits[k]:
                f.write(f"Recall@{k}: {final_recalls[k]:.4f}\n")

    print(f"Results saved to experiments/evaluation_turn_by_turn.txt")
    
    # Print comparison table
    r1 = final_recalls.get(1, 0)
    r10 = final_recalls.get(10, 0)
    r50 = final_recalls.get(50, 0)
    
    print("\nComparison Table:")
    print("Method          | R@1   | R@10  | R@50")
    print("-" * 42)
    print("My full conv    | 0.190 | 0.570 | 0.745")
    print(f"My turn by turn | {r1:.3f} | {r10:.3f} | {r50:.3f}")
    print("KBRD paper      | 0.031 | 0.150 | 0.336")
    print("TPLM paper      | 0.059 | 0.232 | 0.471")


if __name__ == "__main__":
    evaluate(max_samples=200)