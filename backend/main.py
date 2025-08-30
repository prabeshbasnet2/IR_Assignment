from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from search_engine import search, build_index
from classifier import classify_text
from apscheduler.schedulers.background import BackgroundScheduler

app = FastAPI(title="Mini Google Scholar Backend")

# ---------- CORS CONFIGURATION ----------
origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- ROOT ENDPOINT ----------
@app.get("/")
def read_root():
    return {"message": "Information Retrieval + Classification API"}

# ---------- SEARCH ENDPOINT ----------
@app.get("/search")
def search_publications(query: str = '', page: int = 1, size: int = 30):
    result = search(query=query, top_k=100)
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_data = result['data'][start_idx:end_idx]
    return {
        "results": paginated_data,
        "page": page,
        "size": size,
        "totalData": result['totalData'],
    }

# ---------- CLASSIFICATION ENDPOINT ----------
@app.post("/classify")
def classify_document(payload: dict):
    text = payload.get("text", "")
    result = classify_text(text)
    return result

# ---------- SCHEDULER ----------
scheduler = BackgroundScheduler()
# Run build_index every Monday at 2 AM
scheduler.add_job(build_index, "cron", day_of_week="mon", hour=2, minute=0)
scheduler.start()

# Ensure scheduler shuts down cleanly
import atexit
atexit.register(lambda: scheduler.shutdown())
