import feedparser
import json
import traceback

class RSSscraper:
    """
    This script scrapes the RSS of articles around the globe and outputs metadata of these articles in a json file
    """
    def __init__(self, rss_feeds, keywords, output_file="science_news_rss.json"):
        self.rss_feeds = rss_feeds
        self.keywords = keywords
        self.output_file = output_file
        self.articles = []

    def fetch_articles(self):
        count = 0
        for source, url in self.rss_feeds.items():
            try:
                print(f"Fetching RSS feed from: {url}")
                feed = feedparser.parse(url)

                if not feed.entries:
                    print(f"Warning: No entries found for {source}. Skipping.")
                    continue

                for entry in feed.entries[:50]:  # Limit to 50 articles per source
                    try:
                        title = entry.get("title", "No title").strip()
                        link = entry.get("link", "#")
                        summary = entry.get("summary", "No summary available").strip()

                        if any(keyword in title.lower() or keyword in summary.lower() for keyword in self.keywords):
                            self.articles.append({"source": source, "title": title, "link": link, "summary": summary})
                            count += 1
                            print(f"Collected {count} articles so far...")
                    except Exception as e:
                        print(f"Error processing an article from {source}: {e}")
                        traceback.print_exc()
            except Exception as e:
                print(f"Error fetching/parsing RSS feed from {source}: {e}")
                traceback.print_exc()

    def save_articles(self):
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(self.articles, f, ensure_ascii=False, indent=4)
            print(f"Scraped {len(self.articles)} science and health news articles and saved to {self.output_file}.")
        except Exception as e:
            print(f"Error saving articles to JSON: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    rss_feeds = {
       "BBC Science & Health": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "CNN Health": "https://rss.cnn.com/rss/edition_health.rss",
        "NY Times Science": "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
        "Reuters Health": "https://www.reutersagency.com/feed/?best-sectors=health",
        "NPR Health": "https://feeds.npr.org/1128/rss.xml",
        "The Guardian Science": "https://www.theguardian.com/science/rss",
        "Science Magazine": "https://www.sciencemag.org/rss/news_current.xml",
        "Nature News": "https://www.nature.com/subjects/news.rss",
        "Scientific American": "https://www.scientificamerican.com/feed/",
        "Medscape News": "https://www.medscape.com/rss/public/health.xml",
        "MIT Technology Review": "https://www.technologyreview.com/feed/",
        "STAT News Health": "https://www.statnews.com/feed/",
        "The Lancet Health": "https://www.thelancet.com/rssfeed/health.xml",
        "WHO News": "https://www.who.int/rss-feeds/news-english.xml",
        "CDC Newsroom": "https://tools.cdc.gov/api/v2/resources/media/403372.rss",
        "Healthline News": "https://www.healthline.com/rss",
        "Live Science": "https://www.livescience.com/feeds/all",
        "New Scientist": "https://www.newscientist.com/subject/health/feed/",
        "Futurism Science": "https://futurism.com/feed/",
        "Medical News Today": "https://www.medicalnewstoday.com/rss",
        "Harvard Health": "https://www.health.harvard.edu/blog/feed",
        "Psychology Today Science": "https://www.psychologytoday.com/us/front/feed",
        "Science Daily": "https://www.sciencedaily.com/rss/all.xml",
        "National Geographic Science": "https://www.nationalgeographic.com/science/rss/",
        "Fast Company Health": "https://www.fastcompany.com/health/feed/",
        "TIME Health": "https://time.com/section/health/feed/",
        "The Conversation Health": "https://theconversation.com/global/health/rss",
        "Vox Future Perfect": "https://www.vox.com/future-perfect/rss",
        "The Atlantic Science": "https://www.theatlantic.com/feed/category/science/",
        "Smithsonian Magazine Science": "https://www.smithsonianmag.com/rss/latest_articles/",
        "Popular Science": "https://www.popsci.com/rss.xml",
        "Discover Magazine": "https://www.discovermagazine.com/feeds/all",
        "World Health Organization Updates": "https://www.who.int/rss-feeds/news-english.xml",
        "European Medicines Agency": "https://www.ema.europa.eu/en/news-events/rss.xml",
        "Science News": "https://www.sciencenews.org/feed",
        "American Medical Association": "https://www.ama-assn.org/ama-news/rss.xml",
        "Biotech Now": "https://www.bio.org/blog/feed.xml",
        "Genetic Engineering News": "https://www.genengnews.com/feed/",
        "MedPage Today": "https://www.medpagetoday.com/rss.xml",
        "Global Health Now": "https://globalhealthnow.org/rss.xml",
        "Johns Hopkins Health Alerts": "https://www.hopkinsmedicine.org/health/rss.xml",
        "TED Health": "https://www.ted.com/themes/rss/id/5",
        "University of Oxford Science Blog": "https://www.ox.ac.uk/news/science/rss.xml",
        "Stanford Medicine News": "https://med.stanford.edu/news/feed.rss",
        "Yale Health News": "https://news.yale.edu/rss.xml",
        "US National Library of Medicine": "https://www.ncbi.nlm.nih.gov/news/rss.xml",
        "Medical Xpress": "https://medicalxpress.com/rss-feed/",
        "Phys.org Health & Medicine": "https://phys.org/rss-feed/health-news/",
        "Scientific Reports": "https://www.nature.com/srep.rss",
        "Environmental Health Perspectives": "https://ehp.niehs.nih.gov/rss.xml",
        "Food and Drug Administration (FDA) News": "https://www.fda.gov/rss.xml",
        "BBC Science & Health": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
        "CNN Health": "https://rss.cnn.com/rss/edition_health.rss",
        "NY Times Science": "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
        "Reuters Health": "https://www.reutersagency.com/feed/?best-sectors=health",
        "NPR Health": "https://feeds.npr.org/1128/rss.xml",
        "The Guardian Science": "https://www.theguardian.com/science/rss",
        "Science Magazine": "https://www.sciencemag.org/rss/news_current.xml",
        "Nature News": "https://www.nature.com/subjects/news.rss",
        "Scientific American": "https://www.scientificamerican.com/feed/",
        "Medscape News": "https://www.medscape.com/rss/public/health.xml",
        "MIT Technology Review": "https://www.technologyreview.com/feed/",
        "STAT News Health": "https://www.statnews.com/feed/",
        "The Lancet Health": "https://www.thelancet.com/rssfeed/health.xml",
        "WHO News": "https://www.who.int/rss-feeds/news-english.xml",
        "CDC Newsroom": "https://tools.cdc.gov/api/v2/resources/media/403372.rss",
        "Healthline News": "https://www.healthline.com/rss",
        "Live Science": "https://www.livescience.com/feeds/all",
        "New Scientist": "https://www.newscientist.com/subject/health/feed/",
        "Futurism Science": "https://futurism.com/feed/",
        "Medical News Today": "https://www.medicalnewstoday.com/rss",
        "Harvard Health": "https://www.health.harvard.edu/blog/feed",
        "Psychology Today Science": "https://www.psychologytoday.com/us/front/feed",
        "Science Daily": "https://www.sciencedaily.com/rss/all.xml",
        "National Geographic Science": "https://www.nationalgeographic.com/science/rss/",
        "Fast Company Health": "https://www.fastcompany.com/health/feed/",
        "TIME Health": "https://time.com/section/health/feed/",
        "The Conversation Health": "https://theconversation.com/global/health/rss",
        "Vox Future Perfect": "https://www.vox.com/future-perfect/rss",
        "The Atlantic Science": "https://www.theatlantic.com/feed/category/science/"
    }
    
    science_health_keywords = [
        "science", "health", "medicine", "vaccines", "genetics", "biotechnology", "public health", "climate change",
        "medical research", "disease", "mental health", "pandemic", "AI in healthcare", "space exploration", "nutrition",
        "epidemiology", "neuroscience", "longevity", "biomedicine", "scientific discovery", "alternative medicine",
        "pharmaceuticals", "hospital", "medical ethics", "clinical trials", "global health"
    ]
    
    scraper = RSSscraper(rss_feeds, science_health_keywords)
    scraper.fetch_articles()
    scraper.save_articles()
