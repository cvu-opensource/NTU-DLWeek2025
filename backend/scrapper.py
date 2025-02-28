import feedparser
import json
import traceback

class RSSscraper:
    """
    This script scrapes the RSS of articles around the globe for political keywords and outputs metadata of these articles in a json file
    """
    def __init__(self, rss_feeds, keywords, output_file="political_news_rss.json"):
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

                for entry in feed.entries[:10]:  # Limit to 10 articles per source
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
            print(f"Scraped {len(self.articles)} political news articles and saved to {self.output_file}.")
        except Exception as e:
            print(f"Error saving articles to JSON: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    rss_feeds = {  
    "BBC Politics": "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "CNN Politics": "https://rss.cnn.com/rss/cnn_allpolitics.rss",
    "The Guardian Politics": "https://www.theguardian.com/politics/rss",
    "NY Times Politics": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
    "Reuters Politics": "https://www.reutersagency.com/feed/?best-sectors=politics",
    "Politico": "https://www.politico.com/rss/politics.xml",
    "Washington Post Politics": "https://feeds.washingtonpost.com/rss/politics",
    "ABC News Politics": "https://abcnews.go.com/abcnews/politicsheadlines",
    "NBC News Politics": "https://www.nbcnews.com/politics",
    "The Atlantic Politics": "https://www.theatlantic.com/feed/channel/politics/",
    "NPR Politics": "https://feeds.npr.org/1014/rss.xml",
    "Slate Politics": "https://slate.com/feeds/news-and-politics.rss",
    "The Hill": "https://thehill.com/rss/syndicator/19110",
    "Axios Politics": "https://www.axios.com/feeds/politics.xml",
    "RealClearPolitics": "https://www.realclearpolitics.com/index.xml",
    "National Review Politics": "https://www.nationalreview.com/rss.xml",
    "AP News Politics": "https://apnews.com/hub/politics",
    "Time Politics": "https://time.com/feed/politics/",
    "ProPublica": "https://www.propublica.org/feeds/",
    "The Nation Politics": "https://www.thenation.com/feeds/?post_type=article&subject=politics",
    "PBS NewsHour Politics": "https://www.pbs.org/newshour/feeds/rss/politics",
    "Washington Times Politics": "https://www.washingtontimes.com/rss/headlines/news/politics/",
    "The Texas Tribune Politics": "https://www.texastribune.org/feeds/",
    "CTV News Politics": "https://www.ctvnews.ca/rss/ctvnews-politics-public-rss-1.822872",
    "Financial Times World Politics": "https://www.ft.com/world/rss",
    "The Economist Politics": "https://www.economist.com/latest/rss.xml",
    "New York Post Politics": "https://nypost.com/news/feed/",
    "Chicago Tribune Politics": "https://www.chicagotribune.com/politics/rss2.0.xml",
    "Los Angeles Times Politics": "https://www.latimes.com/politics/rss2.0.xml",
    "Bloomberg Politics": "https://www.bloomberg.com/politics/rss",
    "Vox Politics": "https://www.vox.com/rss/index.xml",
    "HuffPost Politics": "https://www.huffpost.com/section/politics/feed",
    "The Verge Policy & Politics": "https://www.theverge.com/policy/rss/index.xml",
    "The Intercept Politics": "https://theintercept.com/feed/?rss",
    "Mother Jones Politics": "https://www.motherjones.com/rss/",
    "USA Today Politics": "https://rssfeeds.usatoday.com/usatodaycomwashington-topstories",
    "Newsweek Politics": "https://www.newsweek.com/politics/feed",
    "Forbes Politics": "https://www.forbes.com/politics/feed/",
    "New Republic Politics": "https://newrepublic.com/rss.xml",
    "Al Jazeera Politics": "https://www.aljazeera.com/xml/rss/all.xml",
    "Christian Science Monitor Politics": "https://rss.csmonitor.com/feeds/politics",
    "Fox News Politics": "https://moxie.foxnews.com/google-publisher/politics.xml",
    "The Daily Beast Politics": "https://feeds.thedailybeast.com/rss/articles",
    "Daily Mail Politics": "https://www.dailymail.co.uk/news/politics/index.rss",
    "The Independent Politics": "https://www.independent.co.uk/news/uk/politics/rss",
    "The Telegraph Politics": "https://www.telegraph.co.uk/politics/rss.xml",
    "Sky News Politics": "https://feeds.skynews.com/feeds/rss/politics.xml",
    "Japan Times Politics": "https://www.japantimes.co.jp/news/politics/feed/",
    "South China Morning Post Politics": "https://www.scmp.com/rss/91/feed",
    "India Today Politics": "https://www.indiatoday.in/rss/home",
    "The Hindu Politics": "https://www.thehindu.com/news/national/?service=rss",
    "Le Monde Politique": "https://www.lemonde.fr/politique/rss_full.xml",
    "Deutsche Welle Politics": "https://rss.dw.com/rdf/rss-en-all",
    "El País Politica": "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/politica",
    "The Moscow Times Politics": "https://www.themoscowtimes.com/rss/news",
    "EU Observer Politics": "https://euobserver.com/rss.xml",
    "Brussels Times Politics": "https://www.brusselstimes.com/feed/",
    "The Local Germany Politics": "https://www.thelocal.de/rss",
    "France 24 Politics": "https://www.france24.com/en/tag/politics/rss",
    "Euronews Politics": "https://www.euronews.com/rss?level=theme&name=politics",
    "The Times of Israel Politics": "https://www.timesofisrael.com/feed/",
    "Middle East Eye Politics": "https://www.middleeasteye.net/rss",
    "Arab News Politics": "https://www.arabnews.com/taxonomy/term/23/feed",
    "South African News24 Politics": "https://www.news24.com/rss?category=Politics",
    "All Africa Politics": "https://allafrica.com/tools/headlines/rdf/politics/headlines.rdf",
    "The Hindu International Politics": "https://www.thehindu.com/news/international/?service=rss",
    "The Korea Herald Politics": "https://www.koreaherald.com/rss/national_politics.xml",
    "Japan Today Politics": "https://japantoday.com/feed",
    "The Jakarta Post Politics": "https://rss.jakpost.net/rss/politics.xml",
    "Manila Bulletin Politics": "https://mb.com.ph/category/news/nation/feed/",
    "The Sydney Morning Herald Politics": "https://www.smh.com.au/rss/politics.xml",
    "The Australian Politics": "https://www.theaustralian.com.au/politics/rss",
    "NZ Herald Politics": "https://www.nzherald.co.nz/rss/topic/politics/",
    "Buenos Aires Times Politics": "https://www.batimes.com.ar/rss/politics.xml",
    "Folha de S.Paulo Politics (Brazil)": "https://www1.folha.uol.com.br/rss/poder.xml",
    "La Nación Politics (Argentina)": "https://www.lanacion.com.ar/rss/tema/politica-t58244",
    "El Universal Mexico Politics": "https://www.eluniversal.com.mx/rss/politica.xml",
    "The Canadian Press Politics": "https://www.cp24.com/rss/canadian-politics",
    "The Toronto Star Politics": "https://www.thestar.com/politics.feed",
    "Daily Sabah Politics (Turkey)": "https://www.dailysabah.com/rss/politics.xml",
    "TASS Russia Politics": "https://tass.com/rss/v2.xml",
    "The Moscow Times Politics": "https://www.themoscowtimes.com/rss/news",
    "Al Jazeera Balkans Politics": "https://balkans.aljazeera.net/feed",
    "Swiss Info Politics": "https://www.swissinfo.ch/eng/politics/rss",
    "Irish Times Politics": "https://www.irishtimes.com/cmlink/news-politics-1.1319192",
    "Hindustan Times Politics": "https://www.hindustantimes.com/rss/india-news/rssfeed.xml",
    "The Straits Times Politics (Singapore)": "https://www.straitstimes.com/news/singapore/politics/rss.xml",
    "Gulf News Politics": "https://gulfnews.com/rss/1.576",
    "Nigeria Guardian Politics": "https://guardian.ng/category/politics/feed/",
    "The East African Politics": "https://www.theeastafrican.co.ke/rss/2710370-3264762-t21oex/index.xml",
    "The Namibian Politics": "https://www.namibian.com.na/rss/Politics",
    "The Standard Kenya Politics": "https://www.standardmedia.co.ke/rss/politics.xml",
    "Daily Nation Politics (Kenya)": "https://www.nation.co.ke/news/politics/rss.xml",
    "The Herald Zimbabwe Politics": "https://www.herald.co.zw/feed/",
    "The Fiji Times Politics": "https://www.fijitimes.com/category/politics/feed/",
    "Bangkok Post Politics": "https://www.bangkokpost.com/rss/data/politics.xml",
    "Vietnam News Politics": "https://vietnamnews.vn/rss/politics-laws.rss",
    "The Cambodia Daily Politics": "https://www.cambodiadaily.com/feed/",
    "Mongolia News Politics": "https://www.mongolia-news.com/rss/category/politics.xml",
    "The Bangkok Post Politics": "https://www.bangkokpost.com/rss/politics",
    "The Philippine Star Politics": "https://www.philstar.com/rss/headlines/politics",
    "Jakarta Globe Politics": "https://jakartaglobe.id/news/politics/rss",
    "The China Post Politics": "https://chinapost.nownews.com/rss",
    "The People's Daily Politics (China)": "http://en.people.cn/rss.xml",
    "SABC News Politics (South Africa)": "https://www.sabcnews.com/sabcnews/category/politics/feed/",
    "Mail & Guardian Politics (South Africa)": "https://mg.co.za/section/politics/feed/",
    "Tanzania Daily News Politics": "https://dailynews.co.tz/rss/category/politics.xml",
    "World Politics Review": "https://www.worldpoliticsreview.com/feed/",
    "The Diplomat Politics": "https://thediplomat.com/feed/",
    "Foreign Policy Politics": "https://foreignpolicy.com/feed/",
    "Council on Foreign Relations Politics": "https://www.cfr.org/rss/global_conflict.xml",
    "International Politics and Society": "https://www.ips-journal.eu/rss.xml",
    "FiveThirtyEight Politics": "https://fivethirtyeight.com/politics/feed/",
    "The American Conservative Politics": "https://www.theamericanconservative.com/feed/",
    "Reason Magazine Politics": "https://reason.com/politics/feed/",
    "Talking Points Memo": "https://talkingpointsmemo.com/feed/",
    "Crooked Media Politics": "https://crooked.com/feed/",
    "Truthout Politics": "https://truthout.org/feed/",
    "The Federalist Politics": "https://thefederalist.com/feed/",
    "American Prospect Politics": "https://prospect.org/feed/",
    "Maclean's Politics": "https://www.macleans.ca/politics/feed/",
    "iPolitics Canada": "https://ipolitics.ca/feed/",
    "Globe and Mail Politics": "https://www.theglobeandmail.com/politics/rss/",
    "National Observer Politics": "https://www.nationalobserver.com/rss.xml",
    "New Statesman Politics": "https://www.newstatesman.com/uk-politics/feed",
    "The Spectator Politics": "https://www.spectator.co.uk/feed/",
    "The Conversation UK Politics": "https://theconversation.com/uk/politics/rss",
    "Morning Star UK Politics": "https://morningstaronline.co.uk/rss",
    "Der Spiegel Politics (Germany)": "https://www.spiegel.de/international/topic/politics/rss.xml",
    "Handelsblatt Global Politics (Germany)": "https://www.handelsblatt.com/rss/politik/",
    "Politico Europe Politics": "https://www.politico.eu/feed/",
    "The Local Spain Politics": "https://www.thelocal.es/feed",
    "El Diario Politica (Spain)": "https://www.eldiario.es/rss/",
    "La Repubblica Politica (Italy)": "https://www.repubblica.it/rss/politica/rss2.0.xml",
    "Il Fatto Quotidiano Politica (Italy)": "https://www.ilfattoquotidiano.it/feed/",
    "De Standaard Politiek (Belgium)": "https://www.standaard.be/rssfeed.aspx?section=politiek",
    "The Conversation AU Politics": "https://theconversation.com/au/politics/rss",
    "The Monthly Politics (Australia)": "https://www.themonthly.com.au/rss",
    "Crikey Politics (Australia)": "https://www.crikey.com.au/feed/",
    "Stuff Politics (New Zealand)": "https://www.stuff.co.nz/rss/national/politics",
    "Global Times Politics (China)": "https://www.globaltimes.cn/rss/China.xml",
    "China Daily Politics": "https://www.chinadaily.com.cn/rss/cndaily.xml",
    "The Korea Times Politics": "https://www.koreatimes.co.kr/www/rss/nation.xml",
    "Bangladesh Daily Star Politics": "https://www.thedailystar.net/rss/politics",
    "Dawn Politics (Pakistan)": "https://www.dawn.com/feeds/news.xml",
    "The Express Tribune Politics (Pakistan)": "https://tribune.com.pk/feed/",
    "The Jakarta Post Politics (Indonesia)": "https://www.thejakartapost.com/rss/national.xml",
    "Scroll.in Politics": "https://scroll.in/rss",
    "The Wire Politics": "https://thewire.in/rss",
    "The Print Politics": "https://theprint.in/category/politics/feed/",
    "Caravan Magazine Politics": "https://caravanmagazine.in/rss.xml",
    "Outlook India Politics": "https://www.outlookindia.com/rss/section/politics",
    "Folha de S.Paulo International Politics": "https://www1.folha.uol.com.br/rss/poder.xml",
    "O Globo Politica (Brazil)": "https://oglobo.globo.com/rss.xml",
    "Telesur English Politics": "https://www.telesurenglish.net/rss/news.xml",
    "La Jornada Mexico Politics": "https://www.jornada.com.mx/rss/politica.xml",
    "Clarín Política (Argentina)": "https://www.clarin.com/rss/politica/",
    "El Mercurio Politica (Chile)": "https://www.emol.com/rss/politica.xml",
    "Mail & Guardian Politics (South Africa)": "https://mg.co.za/section/politics/feed/",
    "Daily Maverick Politics (South Africa)": "https://www.dailymaverick.co.za/section/politics/feed/",
    "Pambazuka News Africa Politics": "https://www.pambazuka.org/rss",
    "The East African Politics": "https://www.theeastafrican.co.ke/rss/2710370-3264762-t21oex/index.xml",
    "GhanaWeb Politics": "https://www.ghanaweb.com/rss/category.php?cid=2",
    "The Citizen Tanzania Politics": "https://www.thecitizen.co.tz/News/Politics/-/1840418/1840418/-/format/xhtml/-/5evdc0z/-/index.xml",
    "Vanguard Politics (Nigeria)": "https://www.vanguardngr.com/category/politics/feed/",
    "Daily Trust Politics (Nigeria)": "https://www.dailytrust.com.ng/rss/politics.xml",
    "RT Politics (Russia)": "https://www.rt.com/rss/",
    "TASS Politics": "https://tass.com/rss/v2.xml",
    }

    political_keywords = [
        "politics", "government", "democracy", "governance", "public policy", "election", 
        "political party", "campaign", "voting", "referendum", "debate", "legislation",
        "constitution", "reform", "president", "prime minister", "congress", "parliament",
        "senator", "governor", "mayor", "house of representatives", "law enforcement",
        "gun control", "climate policy", "foreign affairs", "diplomacy", "NATO", "United Nations",
        "treaty", "sanctions", "military alliance", "geopolitics", "corruption", "political scandal",
        "populism", "fascism", "communism", "socialism", "capitalism", "left-wing", "right-wing",
        "liberalism", "conservatism", "civil rights", "human rights", "media bias", "disinformation",
        "fake news", "press freedom", "protest", "activism", "strike", "civil disobedience",
        "social justice", "feminism", "racial justice", "LGBTQ+ rights"
    ]

    scraper = RSSscraper(rss_feeds, political_keywords)
    scraper.fetch_articles()
    scraper.save_articles()
