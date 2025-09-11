import requests
from lxml import html
import json

def get_url(url, filename):
    # Simple fetch and cache utility
    try:
        with open(filename, 'r', encoding='utf8') as f:
            return f.read()
    except FileNotFoundError:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'w', encoding='utf8') as f:
            f.write(response.text)
        return response.text

def parse(content, mode):
    if mode == 'html':
        return html.fromstring(content)
    elif mode == 'json':
        return json.loads(content)
    else:
        raise ValueError("Mode must be 'html' or 'json'")