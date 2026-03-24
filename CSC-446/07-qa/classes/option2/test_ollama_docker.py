import requests

resp = requests.get("http://localhost:11434/api/tags")
resp.raise_for_status()
data = resp.json()  # {"models": [...]}

for m in data["models"]:
    print(m["name"], m["size"])

