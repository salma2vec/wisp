import requests

r = requests.post("http://localhost:8000/generate", json={
    "prompts": ["Write a haiku about GPUs"],
    "params": {"max_tokens": 32, "temperature": 0.8}
})
print(r.json())
