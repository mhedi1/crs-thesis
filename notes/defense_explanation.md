# Defense Explanation

My project is a conversational movie recommendation system built in three stages.

Stage 1 uses KBRD, a graph-based recommender, to retrieve candidate movies from the dialogue context. I improved this stage by adding stronger entity extraction and metadata enrichment.

Stage 2 uses Qwen3.5:35b as an LLM reranker. Instead of directly generating a movie from nothing, the LLM receives the dialogue and the KBRD candidate list, then selects the most suitable candidate. This makes the system more controlled and easier to evaluate.

Stage 3 uses Qwen again to generate a natural language response explaining the selected recommendation to the user.

For evaluation, I use turn-by-turn ReDial evaluation. Recall@K measures whether the ground-truth movie appears in the KBRD candidate list. Reranker@1 measures whether Qwen selected the correct movie as the final recommendation. I use strict normalized exact title matching to avoid inflated results.
