import re
import ujson
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------- SEARCH PART --------------------

# Download NLTK resources (first time only)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")

# Initialize NLP components
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Globals
dict_search = []
publication_titles = []
publication_authors = []
publication_dates = []
search_texts = []
vectorizer = None
tfidf_matrix = None

def preprocess_text(text: str) -> str:
    """Preprocess text: lowercase, remove special chars, remove stopwords, stem."""
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

def build_index():
    """
    Build the TF-IDF index from publications.json
    """
    global dict_search, publication_titles, publication_authors, publication_dates
    global search_texts, vectorizer, tfidf_matrix

    try:
        file_path = os.path.join(os.path.dirname(__file__), "../crawler/data/publications.json")
        with open(file_path, "r", encoding="utf-8") as doc:
            dict_search = ujson.load(doc)
            publication_titles = [item.get("title", "") for item in dict_search]
            publication_authors = [item.get("authors", []) for item in dict_search]
            publication_dates = [item.get("published_date", item.get("date", "")) for item in dict_search]
            print(f"✅ Loaded {len(dict_search)} documents")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        dict_search = []

    # Preprocess all titles and create searchable text
    search_texts = []
    for i, title in enumerate(publication_titles):
        authors_text = " ".join(publication_authors[i]) if isinstance(publication_authors[i], list) else str(publication_authors[i])
        search_text = f"{title} {authors_text}"
        search_texts.append(preprocess_text(search_text))

    try:
        vectorizer = TfidfVectorizer(min_df=1, max_df=0.8, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(search_texts)
        print(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")
    except Exception as e:
        print(f"❌ Error creating TF-IDF: {e}")

# Run immediately at startup
build_index()

def search(query: str = "", similarity_threshold=0.05, top_k=0, page: int = 1, page_size: int = 10) -> dict:
    """Search publications using TF-IDF + cosine similarity with pagination."""
    if not dict_search or vectorizer is None or tfidf_matrix is None:
        return {"data": [], "totalData": 0, "page": page, "pageSize": page_size, "totalPages": 0}

    if not query or query.strip() == "":
        return {"data": [], "totalData": 0, "page": page, "pageSize": page_size, "totalPages": 0}

    processed_query = preprocess_text(query)
    if not processed_query:
        return {"data": [], "totalData": 0, "page": page, "pageSize": page_size, "totalPages": 0}

    try:
        tfidf_query = vectorizer.transform([processed_query])
    except Exception as e:
        print(f"❌ Query transform failed: {e}")
        return {"data": [], "totalData": 0, "page": page, "pageSize": page_size, "totalPages": 0}

    cosine_similarities = cosine_similarity(tfidf_query, tfidf_matrix)[0]

    results = [(i, score) for i, score in enumerate(cosine_similarities) if score >= similarity_threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    if top_k > 0:
        results = results[:top_k]

    total_results = len(results)
    total_pages = max(1, (total_results + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    paginated_results = results[start:end]

    data = []
    for idx, score in paginated_results:
        item = dict_search[idx].copy()
        item["score"] = round(float(score), 3)
        if "link" not in item:
            item["link"] = f"https://example.com/publication/{idx}"  # fallback
        data.append(item)

    return {
        "data": data,
        "totalData": total_results,
        "page": page,
        "pageSize": page_size,
        "totalPages": total_pages,
    }
