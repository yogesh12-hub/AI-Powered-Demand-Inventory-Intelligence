# Deployment
Not yet deployed or verified on any hosting platform; the steps below are configuration only.

**Local**: `pip install -r requirements.txt && streamlit run app.py` ; API: `uvicorn api.main:app --port 8000` (docs at `/docs`).
**Docker**: `docker build -t foresight .` then `docker run -p 8501:8501 foresight`; for the API override the command: `docker run -p 8000:8000 foresight uvicorn api.main:app --host 0.0.0.0 --port 8000`.
**Streamlit Community Cloud**: point at `app.py`; `requirements.txt` is used automatically. The API needs a separate host (Render, Railway, Cloud Run, etc.).
**Privacy**: uploads live in session memory. Do not commit datasets (`.gitignore` blocks `*.csv` except templates). The API is stateless but has no authentication; add one before exposing it publicly.
