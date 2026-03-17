import requests
import json

API_KEY = "68b4eca8-ad5b-4032-b65d-9336e90d7762"

url = "https://content.guardianapis.com/search"

params = {
    "q": "immigration OR asylum OR migrants OR refugees OR borders",
    "from-date": "2010-01-01",
    "to-date": "2010-01-31",
    "show-fields": "bodyText,wordcount",
    "page-size": 5,
    "api-key": API_KEY
}

response = requests.get(url, params=params)
print(f"Status code: {response.status_code}")
data = response.json()

print(json.dumps(data, indent=2))

results = data["response"]["results"]
print(f"Total results for Jan 2010: {data['response']['total']}")
print(f"\nShowing first {len(results)} articles:\n")

for article in results:
    fields = article.get("fields", {})
    wordcount = fields.get("wordcount", "unknown")
    body_preview = fields.get("bodyText", "")[:200]
    print(f"Title: {article['webTitle']}")
    print(f"Date: {article['webPublicationDate']}")
    print(f"Wordcount: {wordcount}")
    print(f"Body preview: {body_preview}")
    print("---")
