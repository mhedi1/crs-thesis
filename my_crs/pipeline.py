from candidate_generator import generate_candidates
from kbrd_adapter import get_kbrd_candidates
from reranker import rerank
from response_generator import generate_response


def run_case(title: str, history: str, top_k: int = 5, use_kbrd: bool = False) -> None:
    if use_kbrd:
        candidates, _ = get_kbrd_candidates(history, top_k=top_k)
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
User: Animation is fine, something the kids can watch.
""".strip()

    case_3 = """
User: I would like a superhero movie.
System: Do you want something classic or modern?
User: Modern please, something from the last 10 years.
""".strip()

    case_4 = """
User: I love science fiction films.
System: Any particular era or style?
User: I prefer space adventures, something epic.
""".strip()

    case_5 = """
User: I am in the mood for a crime movie.
System: Do you like action crime or psychological thriller?
User: Psychological thriller, something classic and intelligent.
""".strip()

    case_6 = """
User: I want something light and fun to watch tonight.
System: Do you prefer romance or comedy?
User: A romantic comedy would be perfect, nothing too serious.
""".strip()

    case_7 = """
User: I feel like watching an action movie.
System: Any particular decade you prefer?
User: I love 80s action films, something with lots of energy.
""".strip()

    case_8 = """
User: I am not in the mood for fiction tonight.
System: Would you prefer a documentary?
User: Yes, something real and interesting, maybe about nature or society.
""".strip()

    run_case("Old horror", case_1, use_kbrd=True)
    print("\n" + "-" * 70 + "\n")
    run_case("Funny family animation", case_2, use_kbrd=True)
    print("\n" + "-" * 70 + "\n")
    run_case("Modern superhero", case_3, use_kbrd=True)
    print("\n" + "-" * 70 + "\n")
    run_case("Sci-fi space", case_4, use_kbrd=True)
    print("\n" + "-" * 70 + "\n")
    run_case("Classic crime thriller", case_5, use_kbrd=True)
    print("\n" + "-" * 70 + "\n")
    run_case("Romantic comedy", case_6, use_kbrd=True)
    print("\n" + "-" * 70 + "\n")
    run_case("80s action", case_7, use_kbrd=True)
    print("\n" + "-" * 70 + "\n")
    run_case("Documentary", case_8, use_kbrd=True)


if __name__ == "__main__":
    main()