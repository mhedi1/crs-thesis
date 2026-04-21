# Conversational Recommender System (CRS)

This project is a final-year thesis focused on building a conversational recommender system for movies.

## Architecture

The system follows a modular pipeline:

1. Candidate Generation (temporary local generator, later KBRD)
2. Re-ranking using a language model (Qwen)
3. Response generation

## Current Status

- Working prototype implemented
- Candidate generator based on keyword extraction
- Reranker (currently mock, real Qwen integration in progress)
- Response generator working
- KBRD adapter prepared but not fully integrated

## How to run

```bash
python my_crs/pipeline.py