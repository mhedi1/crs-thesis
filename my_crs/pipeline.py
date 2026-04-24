from candidate_generator import generate_candidates
from kbrd_adapter import get_kbrd_candidates
from reranker import rerank
from response_generator import generate_response


def run_case(title: str, history: str, top_k: int = 5, use_kbrd: bool = False) -> None:
    if use_kbrd:
        candidates = get_kbrd_candidates(history, top_k=top_k)
        stage1_name = "KBRD Adapter"
    else:
        candidates = generate_candidates(history, top_k=top_k)
        stage1_name = "Local Candidate Generator"

    selected_movie = rerank(history, candidates)
    response = generate_response(history, selected_movie)

    print("=" * 70)
    print(f"TEST CASE: {title}")
    print(f"Stage 1: {stage1_name}")
    print("\nDialogue:")
    print(history)

    print("\nCandidates:")
    for i, candidate in enumerate(candidates, 1):
        print(f"{i}. {candidate}")

    print("\nSelected movie:")
    print(selected_movie)

    print("\nGenerated response:")
    print(response)
    print("=" * 70)


def main():
    case_1 = """
User: I am looking for a horror film.
System: Did you see A Nightmare on Elm Street?
User: Yes, I have seen it. I want something old.
""".strip()

    case_2 = """
User: I want something funny and family friendly.
System: Do you prefer animation or live action?
User: Animation is fine.
""".strip()

    case_3 = """
User: I would like a superhero movie.
System: Do you want something classic or modern?
User: Modern.
""".strip()

    # First validate full prototype using local stage 1
    run_case("Old horror", case_1, use_kbrd=False)
    #run_case("Funny family animation", case_2, use_kbrd=False)
    #run_case("Modern superhero", case_3, use_kbrd=False)

    # Then optionally test KBRD adapter path
    run_case("KBRD adapter check", case_1, use_kbrd=True)


if __name__ == "__main__":
    main()