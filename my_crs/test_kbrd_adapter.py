from kbrd_adapter import get_kbrd_candidates

dialogue = """
User: I am looking for a horror film.
System: Did you see A Nightmare on Elm Street?
User: Yes, I have seen it. I want something old.
""".strip()

candidates = get_kbrd_candidates(dialogue, top_k=5)

print("KBRD candidates:")
for c in candidates:
    print(c)