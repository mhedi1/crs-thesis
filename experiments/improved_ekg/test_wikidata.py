import requests
import json

query = """
SELECT ?genre ?genreLabel WHERE {
  wd:Q25136484 wdt:P136 ?genre .
  SERVICE wikibase:label { 
    bd:serviceParam wikibase:language "en" . 
  }
}
"""

r = requests.get(
    'https://query.wikidata.org/sparql',
    params={'query': query},
    headers={
        'User-Agent': 'CRSThesisBot/1.0',
        'Accept': 'application/json'
    },
    timeout=30
)

data = r.json()
rows = data['results']['bindings']
print(f'Genres for It (2017): {len(rows)} found')
for row in rows:
    print(' -', row.get('genreLabel', {}).get('value', ''))
