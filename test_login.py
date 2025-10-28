#!/usr/bin/env python3
import requests
import json

# Test login
url = "http://localhost:8000/api/auth/login"
data = {
    "email": "test@example.com",
    "password": "testpass123"
}

print("Testing login endpoint...")
print(f"URL: {url}")
print(f"Data: {json.dumps(data, indent=2)}")
print("\nSending request...")

response = requests.post(url, json=data)

print(f"\nStatus Code: {response.status_code}")
print(f"Headers: {dict(response.headers)}")
print(f"\nResponse Body:")
print(json.dumps(response.json(), indent=2))
