# desktop_ui/utils/api_client.py
import requests

BASE_URL = "http://127.0.0.1:5000"

def get(path, **params):
    return requests.get(f"{BASE_URL}{path}", params=params, timeout=5).json()

def post(path, payload):
    return requests.post(f"{BASE_URL}{path}", json=payload, timeout=5).json()
