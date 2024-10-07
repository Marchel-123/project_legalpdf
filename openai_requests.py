
import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('api_key')

data = {
    "model": "gpt-3.5-turbo",  
    "messages": [{"role": "user", "content": "Say this is a test!"}],
    "temperature": 0.7
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

response = requests.post('https://api.openai.com/v1/chat/completions', json=data, headers=headers)

if response.status_code == 200:
    print(response.json()['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}, {response.text}")
