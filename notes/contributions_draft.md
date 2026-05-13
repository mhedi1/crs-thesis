# Main Contributions

1. Integrated a trained KBRD graph-based recommender into a modern CRS pipeline.

2. Added an LLM reranking stage using Qwen3.5:35b to select the best movie from KBRD candidates.

3. Added Wikidata-based metadata enrichment, including genre, director, and year information.

4. Improved entity extraction and candidate preparation for ReDial-style conversations.

5. Implemented strict turn-by-turn evaluation with Recall@K, MRR, and Reranker@1.

6. Added candidate serialization formats for prompt-format ablation experiments.

7. Built a response generation stage that produces natural language movie recommendations.
