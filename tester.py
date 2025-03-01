import nltk
import spacy
import json
import requests
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from newspaper import Article
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load NLP resources
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# Load sentence embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load pre-trained NLI model
nli_model = pipeline("text-classification", model="roberta-large-mnli")

# Google Custom Search API credentials
GOOGLE_API_KEY = "AIzaSyAniu6Sasz5X3cHGrGMZr87gInURrT08L0"
GOOGLE_CX = "90aa44fa2b12f4722"  # Custom Search Engine ID


### **STEP 1: Extract Key Claims & Named Entities from Base Article** ###
def extract_sentences(text):
    """Splits the article into individual sentences."""
    return nltk.sent_tokenize(text)

def extract_key_entities(text):
    """Extracts named entities (people, places, organizations) from text."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_key_sentences(text, top_n=10):
    """Identifies the most important sentences using TF-IDF ranking."""
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= top_n:
        return sentences  # Return all sentences if fewer than top_n

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    scores = tfidf_matrix.sum(axis=1).A1  # Sum TF-IDF scores per sentence
    ranked_sentences = sorted(zip(scores, sentences), reverse=True)

    return [sentence for _, sentence in ranked_sentences[:top_n]]

def process_text(article):
    """Processes the base article to extract sentences, named entities, and key claims."""
    
    text = article.get("text", "")
    
    # Extract key claims & named entities
    key_sentences = extract_key_sentences(text)
    named_entities = extract_key_entities(text)

    return key_sentences, named_entities


### **STEP 2: Search for Related Articles Using Google API & Extract Text** ###
def search_articles(query, num_results=10):
    """Search Google Custom Search API for news articles related to a given query, excluding base_url."""
    """Search Google Custom Search API for news articles related to a given query."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": GOOGLE_CX,
        "key": GOOGLE_API_KEY,
        "num": num_results
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "items" in data:
            filtered_results = [result["link"] for result in data["items"] if "link" in result]
        return filtered_results
        
    except Exception as e:
        print(f"❌ Error retrieving search results: {e}")
    
    return []

def extract_text_from_url(url):
    """Extracts article text from a given URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"❌ Failed to extract article from {url}: {e}")
        return None

def fetch_reference_articles(query, base_url):
    """Search for related articles and extract their text."""
    retrieved_urls = search_articles(query, num_results=10)
    filtered_urls = [url for url in retrieved_urls if url != base_url]
    retrieved_texts = [extract_text_from_url(url) for url in filtered_urls]
    retrieved_urls, retrieved_texts = zip(*[(url, text) for url, text in zip(retrieved_urls, retrieved_texts) if text]) if retrieved_texts else ([], [])
    return list(retrieved_urls), list(retrieved_texts)


### **STEP 3: Store Articles in a Vector Database** ###
def generate_embedding(text):
    """Converts article text into an embedding vector."""
    if not text:
        print("⚠️ Skipping embedding generation for empty text.")
        return None
    return embedding_model.encode(text, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

def store_vectors(base_embedding, retrieved_embeddings):
    """Stores the base and retrieved article embeddings in FAISS."""
    vector_size = base_embedding.shape[1]
    index = faiss.IndexFlatL2(vector_size)
    all_embeddings = np.vstack([base_embedding] + retrieved_embeddings)
    index.add(all_embeddings)
    return index

def process_vector_storage(base_text, retrieved_texts):
    """Runs embedding generation & vector database storage."""
    
    print("\n🧠 Generating base article embedding...")
    base_embedding = generate_embedding(base_text)
    if base_embedding is None:
        print("❌ Error: Base article embedding could not be generated.")
        return None

    print("📥 Generating embeddings for retrieved articles...")
    retrieved_embeddings = [generate_embedding(text) for text in retrieved_texts if text]
    
    if not retrieved_embeddings:
        print("⚠️ No valid retrieved articles for embedding storage.")
        return None

    print("📥 Storing embeddings in FAISS...")
    index = store_vectors(base_embedding, retrieved_embeddings)
    print(f"✅ Stored {index.ntotal} articles in vector database.")

    return index


### **STEP 4: Detect Misinformation Using Similar Articles from FAISS** ###
def detect_misinformation(base_claims, retrieved_articles):
    """Detects contradictions in the most similar retrieved articles using NLI."""
    results = []
    contradiction_count = 0

    # Ensure valid input
    base_claims = [claim for claim in base_claims if claim and isinstance(claim, str)]
    retrieved_articles = [article for article in retrieved_articles if article and isinstance(article, str)]

    if not base_claims or not retrieved_articles:
        print("❌ No valid claims or articles found.")
        return 0, []

    for claim in base_claims:
        for article in retrieved_articles:
            try:
                # Proper NLI Input Format
                premise = article  # The article text
                hypothesis = f"The article supports the claim: '{claim}'."  # Hypothesis format

                result = nli_model(premise, truncation=True, max_length=512)
                
                # Extract most probable label
                label = result[0]['label']
                score = round(result[0]['score'], 4)  # Confidence score

                results.append({
                    "claim": claim,
                    "retrieved_article_snippet": article[:150] + "...",
                    "classification": label,
                    "confidence": score
                })

                if label == "contradiction":
                    contradiction_count += 1

            except Exception as e:
                print(f"❌ NLI Model Error for Claim: '{claim}': {e}")

    misinformation_score = (contradiction_count / max(1, len(results))) * 100
    return misinformation_score, results

### **STEP 5: Generate Final Report & Verdict** ###
### **STEP 5: Generate Final Report & Verdict** ###
def generate_final_report(misinformation_score, results):
    """Generates a structured report."""
    return json.dumps({"misinformation_score": misinformation_score, "analysis": results}, indent=4)

def generate_final_verdict(misinformation_score, results):
    """Generates a JSON object with misinformation likelihood and contradictions."""
    
    # Extract only contradictions from the results
    contradictions = [
        {"claim": r["claim"], "retrieved_article_snippet": r["retrieved_article_snippet"]}
        for r in results if r["classification"] == "contradiction"
    ]

    # Return structured JSON output
    return {
        "misinformation_likelihood": f"{misinformation_score:.2f}%",
        "contradictions": contradictions
    }


### **RUN FULL PIPELINE** ###
def run_misinformation_pipeline(article):
    """Executes the full misinformation detection pipeline."""
    
    base_url = article.get("url", "")  # Extract base article URL

    print("\n🔍 Step 1: Extracting Key Claims...")
    base_claims, named_entities = process_text(article)

    print("\n🔍 Step 2: Fetching Related Articles...")
    retrieved_urls, retrieved_texts = fetch_reference_articles(article["title"], base_url)

    if not retrieved_texts:
        print("❌ No valid related articles found. Stopping pipeline.")
        return

    print("\n🔍 Step 3: Storing in Vector Database...")
    index = process_vector_storage(article["text"], retrieved_texts)

    if index is None:
        print("❌ Embedding storage failed. Stopping pipeline.")
        return

    # **🔎 Debugging before Step 4**
    print(f"\n🔎 DEBUG: Base Claims: {base_claims}")
    print(f"🔎 DEBUG: Retrieved Articles: {[len(a) if a else 'None' for a in retrieved_texts]}")

    print("\n🔍 Step 4: Running Misinformation Detection...")
    misinformation_score, results = detect_misinformation(base_claims, retrieved_texts)

    #print("\n📌 Final Report:", generate_final_report(misinformation_score, results))
    print("\n📌 Final Verdict:", generate_final_verdict(misinformation_score, results))



### **EXAMPLE RUN** ###
article_input = {
    "source": "BBC Politics",
    "title": "International Development Minister Anneliese Dodds quits over aid cuts",
    "text": "International Development Minister Anneliese Dodds has resigned over the prime minister's cuts to the aid budget. In a letter to Sir Keir Starmer, Dodds said the cuts to international aid, announced earlier this week to fund an increase in defence spending, would \"remove food and healthcare from desperate people - deeply harming the UK's reputation\". She told the PM she had delayed her resignation until after his meeting with President Trump, saying it was \"imperative that you had a united cabinet behind you as you set off for Washington\". The Oxford East MP, who attended cabinet despite not being a cabinet minister, said it was with \"sadness\" that she was resigning. She said that while Sir Keir had been clear he was not \"ideologically opposed\" to international development, the cuts were \"being portrayed as following in President Trump's slipstream of cuts to USAID\". Ahead of his trip to meet the US president, Sir Keir announced aid funding would be reduced from 0.5% of gross national income to 0.3% in 2027 in order to fund an increase in defence spending. In his reply to Dodds's resignation letter, the prime minister thanked the departing minister for her \"hard work, deep commitment and friendship\". He said cutting aid was a \"difficult and painful decision and not one I take lightly\" adding: \"We will do everything we can...to rebuild a capability on development.\" In her resignation letter, Dodds said she welcomed an increase to defence spending at a time when the post-war global order had \"come crashing down\". She added that she understood some of the increase might have to be paid for by cuts to ODA [overseas development assistance]. However, she expressed disappointment that instead of discussing \"our fiscal rules and approach to taxation\", the prime minister had opted to allow the ODA to \"absorb the entire burden\". She said the cuts would \"likely lead to a UK pull-out from numerous African, Caribbean and Western Balkan nations - at a time when Russia has been aggressively increasing its global presence\". \"It will likely lead to withdrawal from regional banks and a reduced commitment to the World Bank; the UK being shut out of numerous multilateral bodies; and a reduced voice for the UK in the G7, G20 and in climate negotiations.\" The spending cuts mean \u00a36bn less will be spent on foreign aid each year. The aid budget is already used to pay for hotels for asylum seekers in the UK, meaning the actual amount spend on aid overseas will be around 0.15% of gross national income. The prime minister's decision to increase defence spending came ahead of his meeting in Washington - the US president has been critical of European countries for not spending enough on defence and instead relying on American military support. He welcomed the UK's commitment to spend more, but Sir Keir has been attacked by international development charities and some of his own MPs for the move. Dodds held off her announcement until the prime minister's return from Washington, in order not to overshadow the crucial visit, and it was clear she did not want to make things difficult for the prime minister. But other MPs have been uneasy about the decision, including Labour MP Sarah Champion, who chairs the international development committee, who said that cutting the aid budget to fund defence spending is a false economy that would \"only make the world less safe\". Labour MP Diane Abbott, who had been critical of the cuts earlier in the week, said it was \"shameful\" that other ministers had not resigned along with Dodds. Dodds's resignation also highlights that decisions the prime minister feels he has to take will be at odds with some of the views of Labour MPs, and those will add to tensions between the leadership and backbenchers. In a post on X, Conservative leader Kemi Badenoch said: \"I disagree with the PM on many things but on reducing the foreign aid budget to fund UK defence? He's absolutely right. \"He may not be able to convince the ministers in his own cabinet, but on this subject, I will back him.\" However one of her backbenchers - and a former international development minister - Andrew Mitchell backed Dodds, accusing Labour of trying \"disgraceful and cynical actions\". \"Shame on them and kudos to a politician of decency and principle,\" he added. Liberal Democrat international development spokesperson Monica Harding said Dodds had \"done the right thing\" describing the government's position as \"unsustainable. She said it was right to increase defence spending but added that \"doing so by cutting the international aid budget is like robbing Peter to pay Paul\". \"Where we withdraw our aid, it's Russia and China who will fill the vacuum.\" Deputy Prime Minister Angela Rayner said she was \"sorry to hear\" of Dodds's resignation. \"It is a really difficult decision that was made but it was absolutely right the PM and cabinet endorse the PM's actions to spend more money on defence,\" she said. Dodds first became a Labour MP in 2017 when she was elected to represent the Oxford East constituency. Under Jeremy Corbyn's leadership of the Labour Party she served as a shadow Treasury minister and was promoted to shadow chancellor when Sir Keir took over. Following Labour's poor performance in the 2021 local elections, she was demoted to the women and equalities brief. Since July 2024, she has served as international development minister. Dodds becomes the fourth minister to leave Starmer's government, following Louise Haigh, Tulip Siddiq and Andrew Gwynne. Some Labour MPs are unhappy about Tory defector Natalie Elphicke's past comments. The BBC chairman helped him secure a loan guarantee weeks before the then-PM recommended him for the role, a paper says. The minister is accused of using demeaning and intimidating language towards a civil servant when he was defence secretary. Eddie Reeves has been the Conservatives' leader on Oxfordshire County Council since May 2021. Labour chair Anneliese Dodds demands answers over \u00a3450,000 donation from ex-Tory treasurer Sir Ehud Sheleg. Copyright 2025 BBC. All rights reserved.\u00a0\u00a0The BBC is not responsible for the content of external sites.\u00a0Read about our approach to external linking. ",
    "url": "https://www.bbc.com/news/articles/cpv44982jlgo"
}
run_misinformation_pipeline(article_input)
