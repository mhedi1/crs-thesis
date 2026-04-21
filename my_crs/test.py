import requests

response = requests.post(
    "http://sinbad2ia.ujaen.es:8050/api/chat",
    json={
        "model": "qwen3.5:35b",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    },
    timeout=120
)

print(response.json())