import requests

r = requests.get("http://sinbad2ia.ujaen.es:8050/api/tags", timeout=30)
r.raise_for_status()

data = r.json()
for m in data.get("models", []):
    print(m["name"])