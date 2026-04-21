import os
import sys
import torch
import pickle
import re
from typing import List, Dict, Any

# -------------------------------------------------------------------------
# CONSTANTS & SETUP
# -------------------------------------------------------------------------
# Robust path resolution targeting the KBRD baseline repository
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Resolves to: thesis_crs/baseline_repo/KBRD_project/KBRD
KBRD_REPO_PATH = os.path.normpath(os.path.join(CURRENT_DIR, "..", "baseline_repo", "KBRD_project", "KBRD"))

if KBRD_REPO_PATH not in sys.path:
    # Safely inject KBRD root so imports like `parlai.agents.kbrd...` work
    sys.path.append(KBRD_REPO_PATH)

# We use global variables to ensure model/data loads lazily exactly once.
_model = None
_entity2id = None
_id2entity = None
_movie_ids = None
_kg = None
_has_error = False # Flag to immediately use fallback logic downstream

def get_fallback_candidates(top_k: int) -> List[Dict[str, Any]]:
    """Safe fallback candidate list returned in case of any failure."""
    mock_candidates = [
        {"id": 101, "title": "The Matrix (1999)", "genre": "Sci-Fi", "decade": "1990s"},
        {"id": 102, "title": "The Exorcist (1973)", "genre": "Horror", "decade": "1970s"},
        {"id": 103, "title": "Shrek (2001)", "genre": "Animation", "decade": "2000s"},
        {"id": 104, "title": "Die Hard (1988)", "genre": "Action", "decade": "1980s"},
        {"id": 105, "title": "The Hangover (2009)", "genre": "Comedy", "decade": "2000s"}
    ]
    return mock_candidates[:top_k]

def _load_kbrd_resources():
    """
    1. LOAD the trained KBRD model and required .pkl data files.
    """
    global _model, _entity2id, _id2entity, _movie_ids, _kg, _has_error
    
    # Skip if already successfully loaded or if we know it's broken
    if _model is not None or _has_error:
        return 
        
    print("[KBRD Adapter] -> STAGE 1: Loading resources and initializing model...")
    
    # Define robust absolute paths
    data_dir = os.path.join(KBRD_REPO_PATH, "data", "redial")
    entity2id_path = os.path.join(data_dir, "entity2entityId.pkl")
    movie_ids_path = os.path.join(data_dir, "movie_ids.pkl")
    kg_path = os.path.join(data_dir, "subkg.pkl")
    model_weights_path = os.path.join(KBRD_REPO_PATH, "saved_model", "kbrd.pt")
    
    # Step 1A: Load Data Dictionaries
    try:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory missing: {data_dir}")
            
        with open(entity2id_path, "rb") as f:
            _entity2id = pickle.load(f)
        with open(movie_ids_path, "rb") as f:
            _movie_ids = pickle.load(f)
        with open(kg_path, "rb") as f:
            _kg = pickle.load(f)
            
        _id2entity = {v: k for k, v in _entity2id.items()}
        print(f"[KBRD Adapter] Successfully loaded data: {len(_entity2id)} entities, {len(_movie_ids)} movies.")
    except Exception as e:
        print(f"[KBRD Adapter ERROR] Failed to load data resources: {e}")
        print("[KBRD Adapter] Reverting to MOCK mode for future interactions.")
        _has_error = True
        return
        
    # Step 1B: Init and Load the PyTorch Model
    try:
        from parlai.agents.kbrd.modules import KBRD
        
        # TODO: Refactor these hardcoded hyperparameters. 
        # In a generic environment, you'd load the original `opt` dict used during training.
        n_relation_fallback = 100 
        dim_fallback = 128
        
        _model = KBRD(
            n_entity=len(_entity2id),
            n_relation=n_relation_fallback, 
            dim=dim_fallback,
            n_hop=2,
            kge_weight=1.0,
            l2_weight=2.5e-6,
            n_memory=32,
            item_update_mode="plus_transform",
            using_all_hops=True,
            kg=_kg,
            entity_kg_emb=None, # Loaded via state_dict downstream
            entity_text_emb=None,
            num_bases=8
        )
        
        if os.path.exists(model_weights_path):
            state_dict = torch.load(model_weights_path, map_location="cpu")
            # Handle ParlAI nested state_dict structures
            model_weights = state_dict.get('model', state_dict)
            # Use strict=False to gracefully load what we can if architectures slightly differ
            missing_keys, unexpected_keys = _model.load_state_dict(model_weights, strict=False)
            if missing_keys or unexpected_keys:
                print(f"[KBRD Adapter WARNING] Model weights mismatch! Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
            else:
                print("[KBRD Adapter] Successfully loaded model weights perfectly.")
        else:
            print(f"[KBRD Adapter WARNING] Model weights file not found at {model_weights_path}. Using uninitialized weights.")
            
        _model.eval() # Ensure dropout and batchnorm components are disabled
        print("[KBRD Adapter] Model initialized and set to evaluation mode.")
    except Exception as e:
        print(f"[KBRD Adapter ERROR] Failed to initialize PyTorch model: {e}")
        _has_error = True
        _model = None

def prepare_input(dialogue: str) -> List[int]:
    """
    2. PREPARE the input:
    Convert dialogue into a list of seed entity IDs expected by KBRD.
    """
    print("[KBRD Adapter] -> STAGE 2: Preparing input from dialogue...")
    if _has_error or not _entity2id:
        print("[KBRD Adapter WARNING] Skipping input preparation due to prior errors.")
        return []
        
    words = re.findall(r"\b[\w-]+\b", dialogue.lower())
    seed_set = []
    
    # Naive entity matcher: checking if the word exists as a DBpedia capitalized resource
    for word in words:
        potential_entity = f"<http://dbpedia.org/resource/{word.capitalize()}>"
        if potential_entity in _entity2id:
            seed_set.append(_entity2id[potential_entity])
            
    if not seed_set:
        print("[KBRD Adapter WARNING] No matching DBpedia entities found in dialogue.")
    else:
        print(f"[KBRD Adapter] Found {len(seed_set)} DBpedia entities linked to dialogue.")
        
    return seed_set

def run_kbrd_model(seed_set: List[int]) -> torch.Tensor:
    """
    3. RUN inference:
    Execute a forward pass with the extracted seed entities.
    """
    print("[KBRD Adapter] -> STAGE 3: Running PyTorch model inference...")
    if _has_error or _model is None:
        print("[KBRD Adapter WARNING] Skipping inference due to prior errors.")
        return torch.zeros(0)
        
    # If seed_set is empty, the model still needs an index list, though performance diminishes
    if not seed_set:
        seed_set = [0] # Supply dummy index 0 to avoid crash if no entities matched
        
    with torch.no_grad():
        try:
            # Batch wrapper since forward expects [batch_1_seeds, batch_2_seeds, ...]
            seed_sets_batch = [seed_set]
            # Dummy labels are mathematically required by forward, but ignored for outputs
            dummy_labels = torch.zeros(1, dtype=torch.long)
            
            return_dict = _model(seed_sets_batch, dummy_labels)
            scores = return_dict["scores"][0] # Extract the 1-D score array for our 1 batch entry
            print(f"[KBRD Adapter] Inference successful. Generated scores matrix of shape {scores.shape}.")
            return scores
        except Exception as e:
            print(f"[KBRD Adapter ERROR] Inference crashed: {e}")
            return torch.zeros(0)

def get_top_k(scores: torch.Tensor, top_k: int) -> List[int]:
    """
    4. EXTRACT top-k recommendations:
    Isolate only valid movies out of all entities and rank them.
    """
    print(f"[KBRD Adapter] -> STAGE 4: Extracting top-{top_k} recommendations...")
    if len(scores) == 0 or _has_error or _movie_ids is None:
        print("[KBRD Adapter WARNING] Skipping extraction due to invalid scores or missing setup.")
        return []
        
    try:
        # Mask out anything that is NOT formally registered as a movie
        movie_scores = scores[torch.LongTensor(_movie_ids)]
        
        # Safely enforce we don't ask for top_k > len(movie_scores)
        k = min(top_k, len(movie_scores))
        if k == 0:
            return []
            
        _, top_indices = torch.topk(movie_scores, k=k)
        
        recommended_entity_ids = []
        for idx in top_indices.tolist():
            actual_entity_id = _movie_ids[idx] # Convert local sub-array idx to global KBRD entity ID
            recommended_entity_ids.append(actual_entity_id)
            
        print(f"[KBRD Adapter] Extracted {len(recommended_entity_ids)} valid recommendations.")
        return recommended_entity_ids
    except Exception as e:
        print(f"[KBRD Adapter ERROR] Failed to extract top_k: {e}")
        return []

def convert_to_candidates(entity_ids: List[int]) -> List[Dict[str, Any]]:
    """
    5. MAP model outputs to usable data:
    Convert indices back to text titles, and ensure Qwen reranker fields exist.
    """
    print("[KBRD Adapter] -> STAGE 5: Formatting output for pipeline compatibility...")
    if not entity_ids or _id2entity is None:
        return []
        
    candidates = []
    for eid in entity_ids:
        entity_uri = _id2entity.get(eid, "")
        
        # We need a clean title string since we might lack an external omdb lookup
        match = re.search(r'resource/(.+)>', entity_uri)
        title = match.group(1).replace("_", " ") if match else f"Unknown Movie {eid}"
        
        candidates.append({
            "id": eid,
            "title": title,
            "genre": "Unknown",  # Default if subkg analysis is unavailable right now
            "decade": "Unknown"  
        })
        
    print(f"[KBRD Adapter] Returning {len(candidates)} fully parsed candidate dictionaries.")
    return candidates

def get_kbrd_candidates(dialogue: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    MAIN ENTRY: Generates candidates for the CRS using the REAL KBRD workflow.
    Ensures safe execution and always falls back to valid structure on any failure.
    """
    print(f"\n{'='*50}\n[KBRD Adapter] Starting Candidate Generation Process\n{'='*50}")
    
    _load_kbrd_resources()
    
    seed_set = prepare_input(dialogue)
    scores = run_kbrd_model(seed_set)
    top_k_entity_ids = get_top_k(scores, top_k)
    candidates = convert_to_candidates(top_k_entity_ids)
    
    if not candidates:
        print("[KBRD Adapter] System ran into an exception or found no candidates. Using SAFE FALLBACK.")
        return get_fallback_candidates(top_k)
        
    return candidates