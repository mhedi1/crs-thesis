# save as check_all.py
import torch
import numpy
import sklearn
import tqdm
import transformers
import sentence_transformers
import spacy
import rapidfuzz
import requests

print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("transformers:", transformers.__version__)
print("sentence_transformers:", sentence_transformers.__version__)
print("spacy:", spacy.__version__)
print("requests:", requests.__version__)
print("CUDA:", torch.cuda.is_available())
print("\nAll packages OK")