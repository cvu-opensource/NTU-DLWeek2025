import json
import requests
from bs4 import BeautifulSoup

# Load the JSON file
input_file = "political_news_rss.json"
output_file = "scraped_news.json"

# Read JSON data
with open(input_file, "r", encoding="utf-8") as file:
    articles = json.load(file)

scraped_data = []

# Function to scrape article text
def scrape_article(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error if the request fails

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = soup.find("title").text if soup.find("title") else "No Title"

        # Extract main text from paragraphs
        paragraphs = soup.find_all("p")
        article_text = " ".join([p.text.strip() for p in paragraphs if p.text])

        return {"title": title, "text": article_text}

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return {"title": "Failed to retrieve", "text": "Error"}

# Iterate over articles and scrape content
for article in articles:
    url = article.get("link")
    if url:
        scraped_info = scrape_article(url)
        scraped_data.append({
            "source": article.get("source", "Unknown"),
            "title": scraped_info["title"],
            "text": scraped_info["text"],
            "url": url
        })

# Save the results to a new JSON file
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(scraped_data, file, indent=4, ensure_ascii=False)

print(f"Scraping completed. Data saved to {output_file}")
