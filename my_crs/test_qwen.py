import requests

url = "http://sinbad2ia.ujaen.es:8050/api/chat"

payload = {
    "model": "qwen3.5:35b",
    "messages": [
        {"role": "user", "content": "Reply with exactly: ANSWER: 1"}
    ],
    "stream": False
}

try:
    response = requests.post(url, json=payload, timeout=180)
    print("Status code:", response.status_code)
    print("Raw response:")
    print(response.text)
except requests.exceptions.RequestException as e:
    print("REQUEST ERROR:")
    print(e)