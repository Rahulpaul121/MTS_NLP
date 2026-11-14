"""
MTS_NLP — FastAPI backend for the Hinglish AI Summarizer.

What it does
------------
Serves the fine-tuned mT5-small model (trained on the 50k Hinglish dataset)
over HTTP. The model itself is NOT stored in this repo or on this machine
permanently — it lives on Google Drive. On first start the API downloads it
(a zip of the model folder) from Drive, unpacks it, caches it on local disk,
and loads it. Later starts reuse the cached copy.

Endpoints
---------
GET  /                     -> service info
GET  /health               -> {"status": "ok", "model_loaded": true, ...}
POST /api/v1/process-chat  -> {"conversation": "..."}  returns
                              {"summary": "...", "truncated": false}

Configuration (environment variables)
-------------------------------------
DRIVE_FILE_ID   Google Drive file id of the model zip (the part of the
                sharing link after /file/d/ and before /view).
                Alternative: DRIVE_FILE_URL with the full sharing link.
MODEL_PATH      Local folder the model is cached in.
                Default: ./models/hinglish_summarizer_50k
ALLOWED_ORIGINS Comma-separated CORS origins. Default: * (fine locally;
                restrict this if you deploy publicly).

Run
---
    pip install -r requirements.txt
    export DRIVE_FILE_ID=<your-model-zip-file-id>
    uvicorn api:app --reload --port 8000
"""

import logging
import os
import shutil
import zipfile
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mts_nlp.api")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "./models/hinglish_summarizer_50k"))
DRIVE_FILE_ID = os.environ.get("DRIVE_FILE_ID", "").strip()
DRIVE_FILE_URL = os.environ.get("DRIVE_FILE_URL", "").strip()
MAX_INPUT_TOKENS = 256          # must match training / main.py
MAX_NEW_TOKENS = 100

_origins = os.environ.get("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if _origins == "*" else [o.strip() for o in _origins.split(",") if o.strip()]
if ALLOWED_ORIGINS == ["*"]:
    logger.warning("CORS is open to all origins (*). Set ALLOWED_ORIGINS if you deploy this publicly.")


# --------------------------------------------------------------------------
# Schemas (same contract as the original main.py, plus a truncation flag)
# --------------------------------------------------------------------------
class TranslationRequest(BaseModel):
    conversation: str = Field(..., description="Hinglish conversation, one 'Name: message' per line")


class TranslationResponse(BaseModel):
    summary: str
    truncated: bool = False


# --------------------------------------------------------------------------
# Model download from Google Drive + loading
# --------------------------------------------------------------------------
def _looks_like_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def ensure_model() -> Path:
    """Make sure the model folder exists locally, downloading it from
    Google Drive (once) if it doesn't. Returns the model folder path."""
    if _looks_like_model_dir(MODEL_PATH):
        logger.info("Using cached model at %s", MODEL_PATH)
        return MODEL_PATH

    file_id = DRIVE_FILE_ID
    if not file_id and DRIVE_FILE_URL:
        # Extract the id from a ".../file/d/<ID>/view?usp=sharing" link.
        parts = DRIVE_FILE_URL.split("/file/d/")
        if len(parts) == 2 and parts[1]:
            file_id = parts[1].split("/")[0]

    if not file_id:
        raise RuntimeError(
            f"Model not found at {MODEL_PATH} and no Google Drive source configured. "
            "Set DRIVE_FILE_ID (or DRIVE_FILE_URL) to the shared model zip on Drive, "
            "or place the model folder at MODEL_PATH. See README.md."
        )

    import gdown  # imported lazily so a cached model never needs it

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    zip_path = MODEL_PATH.parent / "hinglish_summarizer_50k.zip"

    logger.info("Downloading model zip from Google Drive (file id %s) ...", file_id)
    gdown.download(id=file_id, output=str(zip_path), quiet=False)

    logger.info("Extracting model to %s ...", MODEL_PATH)
    tmp_extract = MODEL_PATH.parent / "_model_extract_tmp"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    tmp_extract.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_extract)
    zip_path.unlink(missing_ok=True)

    # The zip may contain the model files at its root, or inside a single
    # top-level folder (e.g. hinglish_summarizer_50k/config.json). Handle both.
    if _looks_like_model_dir(tmp_extract):
        src = tmp_extract
    else:
        candidates = [p for p in tmp_extract.iterdir() if _looks_like_model_dir(p)]
        if not candidates:
            shutil.rmtree(tmp_extract, ignore_errors=True)
            raise RuntimeError("Downloaded zip does not contain a model folder (no config.json found inside).")
        src = candidates[0]

    if MODEL_PATH.exists():
        shutil.rmtree(MODEL_PATH)
    shutil.move(str(src), str(MODEL_PATH))
    shutil.rmtree(tmp_extract, ignore_errors=True)

    if not _looks_like_model_dir(MODEL_PATH):
        raise RuntimeError(f"Model extraction finished but {MODEL_PATH} has no config.json.")
    logger.info("Model ready at %s", MODEL_PATH)
    return MODEL_PATH


class Summarizer:
    """Loads the mT5 model once and generates summaries with the exact
    settings used in training / the original main.py."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.load_error: str | None = None

    def load(self) -> None:
        """Download (first run) + load the model. Runs in a background thread
        at startup so the HTTP port opens immediately — hosts like Render
        require the port to bind quickly, while the ~1.2 GB Drive download
        takes minutes. /health reports progress; the summarize endpoint
        returns 503 until loading finishes."""
        try:
            model_dir = ensure_model()
            logger.info("Loading mT5 model from %s on %s ...", model_dir, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir)).to(self.device)
            self.model.eval()
            logger.info("Model loaded.")
        except Exception as e:  # keep the server up; surface the error via /health
            self.load_error = str(e)
            logger.error("Model failed to load: %s", e)

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def summarize(self, conversation: str) -> tuple[str, bool]:
        # Count tokens WITHOUT truncation first, so long inputs are flagged
        # instead of silently cut (the original main.py truncated silently).
        token_count = len(self.tokenizer(conversation, add_special_tokens=True)["input_ids"])
        truncated = token_count > MAX_INPUT_TOKENS
        if truncated:
            logger.warning("Input is %d tokens; truncating to %d.", token_count, MAX_INPUT_TOKENS)

        inputs = self.tokenizer(
            conversation,
            return_tensors="pt",
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                repetition_penalty=1.2,
                early_stopping=True,
            )
        summary = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return summary, truncated


summarizer = Summarizer()

# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = FastAPI(title="Hinglish AI Summarizer", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    import threading

    threading.Thread(target=summarizer.load, daemon=True).start()


@app.get("/")
def root():
    return {
        "service": "Hinglish AI Summarizer",
        "docs": "/docs",
        "health": "/health",
        "endpoint": "POST /api/v1/process-chat",
        "frontend": "Run the Streamlit app:  streamlit run streamlit_app.py",
    }


@app.get("/health")
def health():
    if summarizer.loaded:
        status = "ok"
    elif summarizer.load_error:
        status = "error"
    else:
        status = "loading"
    return {
        "status": status,
        "model_loaded": summarizer.loaded,
        "load_error": summarizer.load_error,
        "model_path": str(MODEL_PATH),
        "model_host": "google-drive",
        "device": str(summarizer.device),
    }


@app.post("/api/v1/process-chat", response_model=TranslationResponse)
def process_chat(request: TranslationRequest):
    # Empty-check is BEFORE the try block, so the 400 reaches the client
    # as a 400 (in the original main.py it was re-raised as a 500).
    conversation = request.conversation.strip()
    if not conversation:
        raise HTTPException(status_code=400, detail="Conversation text cannot be empty.")
    if not summarizer.loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again in a moment.")

    try:
        summary, truncated = summarizer.summarize(conversation)
    except Exception as e:  # genuine model errors -> 500 with the message
        raise HTTPException(status_code=500, detail=str(e))

    return TranslationResponse(summary=summary, truncated=truncated)
