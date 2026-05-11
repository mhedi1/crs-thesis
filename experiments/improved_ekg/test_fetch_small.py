import pickle
import json
import sys
sys.path.insert(0, 
    r'C:\Users\mhfou\Desktop\thesis_crs\experiments\improved_ekg')

from fetch_metadata import (
    load_movie_entities,
    get_wikidata_ids_batch,
    get_genres_from_wikidata_batch,
    clean_genres
)

print("Loading movies...")
movie_entities = load_movie_entities()
items = list(movie_entities.items())[:10]

print(f"Testing with {len(items)} movies...")
uris = [uri for _, uri in items]
eid_lookup = {uri: eid for eid, uri in items}

print("Step 1: Getting Wikidata IDs from DBpedia...")
wikidata_map = get_wikidata_ids_batch(uris)
print(f"Found {len(wikidata_map)} Wikidata IDs")

qid_to_eid = {}
for dbp_uri, qid in wikidata_map.items():
    eid = eid_lookup.get(dbp_uri)
    if eid is not None:
        qid_to_eid[qid] = eid

print("Step 2: Getting genres from Wikidata...")
genre_data = get_genres_from_wikidata_batch(qid_to_eid)
print(f"Found genre data for {len(genre_data)} movies")

print("\nResults:")
for eid, data in genre_data.items():
    print(f"  Entity {eid}:")
    print(f"    genres_clean: {data.get('genres_clean')}")
    print(f"    directors: {data.get('directors')}")
    print(f"    year: {data.get('year')}")
