---
marp: true
theme: default
class: invert
paginate: true
size: 16:9
style: |
  h1 { color: #4da6ff; font-weight: bold; }
  h2 { color: #80c1ff; border-bottom: 2px solid #333; padding-bottom: 10px; }
  strong { color: #b3d9ff; }
  table { width: 100%; border-collapse: collapse; margin-top: 20px; }
  th { background-color: #004d99; color: white; padding: 15px; }
  td { background-color: #1a1a1a; padding: 15px; border: 1px solid #333; }
  .architecture { display: flex; justify-content: space-between; align-items: center; margin-top: 50px; }
  .box { background: #262626; border-left: 6px solid #4da6ff; padding: 20px; border-radius: 8px; width: 28%; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
  .box h3 { margin-top: 0; color: #4da6ff; }
  .box p { font-size: 0.9em; margin-bottom: 0; color: #ccc; }
  .arrow { font-size: 2em; color: #666; }
---

<!-- _class: lead -->
# Conversational Recommendation System (CRS) for Movies
## Mid-Term Thesis Defense
**Author:** Software Engineering Student
**Date:** May 2026

---

## 1. Introduction & Motivation

- **The Goal:** Build an interactive movie recommendation system that naturally converses with users to discover their preferences.
- **The Challenge:** Recommendation requires both **accurate retrieval** (identifying the right movie) and **engaging conversation** (explaining the choice).
- **The Approach:** A robust, hybrid 3-stage pipeline combining a knowledge-graph-based recommender (**KBRD**) with a powerful Large Language Model (**Qwen3.5:35b**).

---

## 2. Architecture Overview

<div class="architecture">
  <div class="box">
    <h3>1. Retrieval (KBRD)</h3>
    <p>Entity extraction (spaCy, n-grams) & DBpedia linking.<br><br>Retrieves top candidate movies using graph model.</p>
  </div>
  <div class="arrow">➔</div>
  <div class="box">
    <h3>2. Reranking (LLM)</h3>
    <p>Candidates are enriched with Wikidata (genre, year, director).<br><br>Qwen selects the most suitable movie.</p>
  </div>
  <div class="arrow">➔</div>
  <div class="box">
    <h3>3. Generation (LLM)</h3>
    <p>Using the selected movie and dialogue history.<br><br>Qwen generates a natural language response.</p>
  </div>
</div>

---

## 3. Technical Contributions

Based on the implemented pipeline, key contributions include:

1. **Neural Model Integration:** Successfully loaded and integrated a trained ParlAI-based KBRD model into a modern pipeline.
2. **Advanced Reranking:** Replaced generic LLM generation with a controlled **Qwen3.5:35b** reranking stage for verified recommendations.
3. **Robust Entity Extraction:** Implemented hybrid extraction combining **spaCy**, n-grams, and explicit temporal clues (e.g., "1980s").
4. **Metadata Enrichment:** Augmented KBRD candidates with real-time genre, director, and year attributes to guide the LLM.

---

## 4. Evaluation Methodology

To ensure honest, rigorous results, the system employs a **turn-by-turn ReDial evaluation** using strict exact title matching.

- **Recall@K:** Measures if the ground-truth movie appears in KBRD's top K candidates.
- **MRR (Mean Reciprocal Rank):** Evaluates the ranking quality of the KBRD candidate list.
- **Reranker@1:** Measures if the LLM correctly identified the exact ground-truth movie from the candidate list.

*Note: Strict normalized exact matching is used to avoid inflated scores.*

---

## 5. Current Mid-Term Results

Based on the recent ReDial evaluation run (Format 3, 200 conversations, 640 instances):

| Metric | Score | Interpretation |
|---|---|---|
| **Recall@1** | 4.22% | Top KBRD prediction matches exactly. |
| **Recall@10** | 17.97% | True movie is in top 10 candidates. |
| **Recall@50** | 33.59% | True movie is in top 50 candidates. |
| **MRR** | 0.0838 | Average ranking quality. |
| **Reranker@1** | **9.38%** | **LLM correctly selects the movie >2x more often than KBRD alone.** |

---

## 6. Limitations & Challenges

Honest reflection on current constraints:

- **Legacy Dependencies:** The KBRD baseline relies on legacy ParlAI code, requiring careful resource management and strict CUDA constraints.
- **API Latency:** Evaluating with Qwen3.5:35b via an external university API slows down testing and depends on server load.
- **Conservative Evaluation:** Strict exact title matching is academically sound but may undercount valid recommendations (e.g., aliases or sequels).
- **Metric Constraints:** Standard NLP metrics (BLEU, ROUGE) struggle to accurately capture open-ended conversational quality.

---

## 7. Next Steps & Conclusion

### Immediate Roadmap
- Conduct **serialization format ablation** experiments (Formats 1, 2, 4).
- Complete the evaluation of **conversational response generation** (Distinct-N, BLEU, ROUGE).
- Extend testing to the **INSPIRED** dataset.

### Conclusion
We have established a highly modular, verifiable pipeline. By splitting retrieval and reasoning, we combine the wide coverage of Knowledge Graphs with the deep contextual understanding of LLMs.
