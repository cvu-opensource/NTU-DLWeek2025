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
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")


# Google Custom Search API credentials
GOOGLE_API_KEY = "api_key"
GOOGLE_CX = "cx_here"  # Custom Search Engine ID


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
        None
    return []

def extract_text_from_url(url):
    """Extracts article text from a given URL while filtering out ads and unwanted content."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()

        # **Filter criteria**
        ad_keywords = ["Picks for you", "Sign in", "Subscribe", "Enable notifications", "Cookies", "Advertisement", "Continue reading"]
        min_text_length = 200  # Ignore very short texts

        # **Check if the article text contains ads or is too short**
        if any(keyword in text for keyword in ad_keywords) or len(text) < min_text_length:
            return None

        return text

    except Exception as e:
        return None

def fetch_reference_articles(query, base_url):
    """Search for related articles and extract their text."""
    retrieved_urls = search_articles(query, num_results=10)
    filtered_urls = [url for url in retrieved_urls if url != base_url]
    retrieved_texts = [extract_text_from_url(url) for url in filtered_urls]
    retrieved_urls, retrieved_texts = zip(*[(url, text) for url, text in zip(retrieved_urls, retrieved_texts) if text]) if retrieved_texts else ([], [])
    return list(retrieved_urls), list(retrieved_texts)

#step 3
def generate_embedding(text):
    """Converts article text into an embedding vector."""
    if not text:
        return None
    return embedding_model.encode(text, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

def store_vectors(base_embedding, retrieved_embeddings):
    """Stores the base and retrieved article embeddings in FAISS."""
    vector_size = base_embedding.shape[1]
    index = faiss.IndexFlatL2(vector_size)
    all_embeddings = np.vstack([base_embedding] + retrieved_embeddings)
    index.add(all_embeddings)
    return index

def find_similar_articles(index, query_embedding, retrieved_texts, top_k=10):
    """Finds the top_k most similar articles from FAISS index."""
    distances, indices = index.search(query_embedding, top_k)
    similar_articles = [retrieved_texts[i] for i in indices[0] if i < len(retrieved_texts)]
    return similar_articles

def process_vector_storage(base_text, retrieved_texts):
    """Runs embedding generation & vector database storage."""
    
    base_embedding = generate_embedding(base_text)
    if base_embedding is None:
        return None, None

    retrieved_embeddings = [generate_embedding(text) for text in retrieved_texts if text]
    
    if not retrieved_embeddings:
        print("⚠️ No valid retrieved articles for embedding storage.")
        return None, base_embedding

    index = store_vectors(base_embedding, retrieved_embeddings)

    return index, base_embedding

#step 4
def detect_misinformation(base_claims, retrieved_articles, index, base_embedding):
    """Detects contradictions using only the most relevant retrieved articles."""
    results = []
    contradiction_count = 0
    valid_comparisons = 0

    # Find the most similar articles
    relevant_articles = find_similar_articles(index, base_embedding, retrieved_articles, top_k=3)

    for claim in base_claims:
        for article in relevant_articles:  # Only use the most relevant articles
            try:
                premise = article  # Retrieved article text

                result = nli_model(premise, truncation=True, max_length=512)

                label = result[0]['label']
                score = round(result[0]['score'], 4)

                if label == "contradiction":
                    contradiction_count += 1

                if score >= 0.75:
                    valid_comparisons += 1
                    results.append({
                        "claim": claim,
                        "retrieved_article_snippet": article[:150] + "...",
                        "classification": label,
                        "confidence": score
                    })

            except Exception as e:
                None

    misinformation_score = (contradiction_count / max(1, valid_comparisons)) * 100 if valid_comparisons > 0 else 0

    return misinformation_score, results

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

    #Step 1: Extracting Key Claims
    base_claims, named_entities = process_text(article)

    #Step 2: Fetching Related Articles
    retrieved_urls, retrieved_texts = fetch_reference_articles(article["title"], base_url)

    if not retrieved_texts:
        return
    
    #Step 3: Storing in Vector Database
    #index = process_vector_storage(article["text"], retrieved_texts)
    index, base_embedding = process_vector_storage(article["text"], retrieved_texts)


    if index is None:
        return

    #Step 4: Running Misinformation Detection

    misinformation_score, results = detect_misinformation(base_claims, retrieved_texts, index, base_embedding)
    print(generate_final_verdict(misinformation_score, results))
    return generate_final_verdict(misinformation_score, results)


### **EXAMPLE RUN** ###
article_input = {
  "title": "International Development Minister Anneliese Dodds quits over aid cuts",
  "text": "International Development Minister Anneliese Dodds has resigned over the prime minister's cuts to the aid budget.  In a letter to Sir Keir Starmer, Dodds said the cuts to international aid, announced earlier this week to fund an increase in defence spending, would \"remove food and healthcare from desperate people - deeply harming the UK's reputation\".  She told the PM she had delayed her resignation until after his meeting with President Trump, saying it was \"imperative that you had a united cabinet behind you as you set off for Washington\".",
  "source": "bbc.com",
  "author": "None",
  "published_date": "2025-02-28T17:06:10.619Z"
}

run_misinformation_pipeline(article_input)
