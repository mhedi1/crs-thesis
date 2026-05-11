import pickle
import json
import re
import time
import requests
import os

ENTITY2ID_PATH = r"C:\Users\mhfou\Desktop\thesis_crs\baseline_repo\KBRD_project\KBRD\data\redial\entity2entityId.pkl"
OUTPUT_PATH = r"C:\Users\mhfou\Desktop\thesis_crs\experiments\improved_ekg\genre_cache.json"
DBPEDIA_SPARQL = "https://dbpedia.org/sparql"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 3


def clean_genres(genre_labels, max_genres=2):
    """Take first max_genres genres, remove film suffix."""
    cleaned = []
    seen = set()
    for g in genre_labels:
        clean = g.replace(' film', '').strip().title()
        if clean and clean not in seen:
            cleaned.append(clean)
            seen.add(clean)
    return cleaned[:max_genres]


def load_movie_entities():
    with open(ENTITY2ID_PATH, 'rb') as f:
        entity2id = pickle.load(f)
    
    movie_entities = {}
    for uri, eid in entity2id.items():
        if not isinstance(uri, str):
            continue
        clean_uri = uri.replace('<', '').replace('>', '')
        if '_film' in clean_uri.lower() or \
           '_(film)' in clean_uri.lower():
            movie_entities[eid] = clean_uri
    
    print(f"Total entities: {len(entity2id)}")
    print(f"Movie entities: {len(movie_entities)}")
    return movie_entities


def get_wikidata_ids_batch(dbpedia_uris):
    """Get Wikidata IDs from DBpedia sameAs for a batch."""
    uri_values = " ".join([f"<{u}>" for u in dbpedia_uris])
    query = f"""
    SELECT ?movie ?wikidata WHERE {{
        VALUES ?movie {{ {uri_values} }}
        ?movie owl:sameAs ?wikidata .
        FILTER(STRSTARTS(STR(?wikidata),
               "http://www.wikidata.org/entity/Q"))
    }}
    """
    try:
        r = requests.get(
            DBPEDIA_SPARQL,
            params={
                'query': query,
                'format': 'application/sparql-results+json'
            },
            timeout=30
        )
        if r.status_code == 200:
            data = r.json()
            result = {}
            for row in data['results']['bindings']:
                movie = row['movie']['value']
                wikidata = row['wikidata']['value']
                qid = wikidata.split('/')[-1]
                result[movie] = qid
            return result
    except Exception as e:
        print(f"  DBpedia error: {e}")
    return {}


def get_genres_from_wikidata_batch(qid_to_eid):
    """Query Wikidata genres for a batch of QIDs."""
    if not qid_to_eid:
        return {}

    values = " ".join([f"wd:{qid}" for qid in qid_to_eid])
    query = f"""
    SELECT ?movie ?genreLabel ?directorLabel ?year WHERE {{
        VALUES ?movie {{ {values} }}
        OPTIONAL {{ ?movie wdt:P136 ?genre . }}
        OPTIONAL {{ ?movie wdt:P57 ?director . }}
        OPTIONAL {{
            ?movie wdt:P577 ?releaseDate .
            BIND(YEAR(?releaseDate) AS ?year)
        }}
        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en" .
        }}
    }}
    LIMIT 500
    """
    try:
        r = requests.get(
            WIKIDATA_SPARQL,
            params={'query': query},
            headers={
                'User-Agent': 'CRSThesisBot/1.0',
                'Accept': 'application/json'
            },
            timeout=30
        )
        if r.status_code == 200:
            data = r.json()
            results = {}
            for row in data['results']['bindings']:
                qid = row['movie']['value'].split('/')[-1]
                eid = qid_to_eid.get(qid)
                if eid is None:
                    continue
                if eid not in results:
                    results[eid] = {
                        'genres': [],
                        'directors': [],
                        'year': None,
                        'wikidata_id': qid
                    }
                if 'genreLabel' in row:
                    g = row['genreLabel']['value']
                    if g not in results[eid]['genres']:
                        results[eid]['genres'].append(g)
                if 'directorLabel' in row:
                    d = row['directorLabel']['value']
                    if d not in results[eid]['directors']:
                        results[eid]['directors'].append(d)
                if 'year' in row and \
                   results[eid]['year'] is None:
                    results[eid]['year'] = \
                        row['year']['value']
            for eid in results:
                raw = results[eid]['genres']
                results[eid]['genres_clean'] = \
                    clean_genres(raw)
            return results
    except Exception as e:
        print(f"  Wikidata error: {e}")
    return {}


def main():
    print("="*60)
    print("Metadata Fetcher - DBpedia + Wikidata")
    print("="*60)

    cache = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r') as f:
            cache = json.load(f)
        print(f"Loaded cache: {len(cache)} entries")

    movie_entities = load_movie_entities()
    to_fetch = {
        eid: uri for eid, uri in movie_entities.items()
        if str(eid) not in cache
    }
    print(f"To fetch: {len(to_fetch)} movies")

    if not to_fetch:
        print("Cache complete.")
        return

    items = list(to_fetch.items())
    total_batches = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(items))
        batch = items[start:end]

        print(f"Batch {batch_num+1}/{total_batches}...",
              end=' ', flush=True)

        dbpedia_uris = [uri for _, uri in batch]
        eid_lookup = {uri: eid for eid, uri in batch}
        wikidata_map = get_wikidata_ids_batch(dbpedia_uris)

        qid_to_eid = {}
        for dbp_uri, qid in wikidata_map.items():
            eid = eid_lookup.get(dbp_uri)
            if eid is not None:
                qid_to_eid[qid] = eid

        genre_data = get_genres_from_wikidata_batch(qid_to_eid)

        genres_found = 0
        for eid, data in genre_data.items():
            cache[str(eid)] = data
            if data.get('genres_clean'):
                genres_found += 1

        for eid, uri in batch:
            if str(eid) not in cache:
                cache[str(eid)] = {
                    'genres': [],
                    'genres_clean': [],
                    'directors': [],
                    'year': None,
                    'wikidata_id': None
                }

        print(f"genres: {genres_found}/{len(batch)}")

        if (batch_num + 1) % 10 == 0:
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(cache, f, indent=2)
            print(f"  Saved: {len(cache)} entries")

        time.sleep(SLEEP_BETWEEN_BATCHES)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(cache, f, indent=2)

    total = len(cache)
    with_genre = sum(
        1 for v in cache.values()
        if v.get('genres_clean'))
    with_dir = sum(
        1 for v in cache.values()
        if v.get('directors'))
    with_year = sum(
        1 for v in cache.values()
        if v.get('year'))

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Total: {total}")
    print(f"With genre:    {with_genre} "
          f"({100*with_genre//total if total else 0}%)")
    print(f"With director: {with_dir} "
          f"({100*with_dir//total if total else 0}%)")
    print(f"With year:     {with_year} "
          f"({100*with_year//total if total else 0}%)")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
