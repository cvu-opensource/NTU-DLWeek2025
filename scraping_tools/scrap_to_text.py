import json
import requests
from bs4 import BeautifulSoup
import time

class LinkToTextScraper:
    """
    This script takes the links of articles provided in the json file and scrapes them for the text of the article, outputting them in a new json file 
    """
    def __init__(self, input_file="science_news_rss.json", output_file="science_scraped_news.json"):
        self.input_file = input_file
        self.output_file = output_file
        self.articles = []
        self.scraped_data = []

    def load_articles(self):
        with open(self.input_file, "r", encoding="utf-8") as file:
            self.articles = json.load(file)

    def scrape_article(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.find("title").text if soup.find("title") else "No Title"
            paragraphs = soup.find_all("p")
            article_text = " ".join([p.text.strip() for p in paragraphs if p.text])
            
            return {"title": title, "text": article_text}
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            return {"title": "Failed to retrieve", "text": "Error"}

    def scrape_articles(self):
        count = 0
        for article in self.articles:
            url = article.get("link")
            if url:
                scraped_info = self.scrape_article(url)
                self.scraped_data.append({
                    "source": article.get("source", "Unknown"),
                    "title": scraped_info["title"],
                    "text": scraped_info["text"],
                    "url": url
                })
                count += 1
                print(count, "Count")
                time.sleep(2)

    def save_scraped_data(self):
        with open(self.output_file, "w", encoding="utf-8") as file:
            json.dump(self.scraped_data, file, indent=4, ensure_ascii=False)
        print(f"Scraping completed. Data saved to {self.output_file}")

if __name__ == "__main__":
    scraper = LinkToTextScraper()
    scraper.load_articles()
    scraper.scrape_articles()
    scraper.save_scraped_data()
