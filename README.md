# MTS_NLP — Hinglish AI Summarizer (FastAPI + Streamlit, model on Google Drive)

A Hinglish conversation goes in, an English summary comes out — using the
mT5-small model fine-tuned on the 50,000-row Hinglish dataset.

The app is split into three pieces:

```
┌────────────────────┐   POST /api/v1/process-chat   ┌────────────────────┐
│  Streamlit frontend │ ───────────────────────────▶ │   FastAPI backend   │
│  streamlit_app.py   │ ◀─────────────────────────── │      api.py         │
└────────────────────┘        {"summary": "..."}      └─────────┬──────────┘
                                                                │ first start only:
                                                                │ download model zip
                                                                ▼
                                                      ┌────────────────────┐
                                                      │    Google Drive     │
                                                      │ hinglish_summarizer │
                                                      │   _50k.zip (model)  │
                                                      └────────────────────┘
```

The model (~1.2 GB) is **never committed to Git and never stored in this
folder** — it stays on Google Drive. The API downloads and caches it on
first start; later starts reuse the cache.

## Files

| File | What it is |
|---|---|
| `api.py` | FastAPI backend. Downloads the model from Drive, loads mT5, serves the summary endpoint. |
| `streamlit_app.py` | Streamlit frontend. Collects the conversation, calls the API, shows the summary. |
| `requirements.txt` | All Python dependencies. |
| `.env.example` | Configuration template (Drive file id, ports, etc.). |

## Step 1 — Put the model zip on Google Drive (one time)

After training in Colab, your model is in Drive at
`MyDrive/MTS_NLP_models/hinglish_summarizer_50k/`. Zip the **contents** of
that folder and share the zip. In Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile
src = '/content/drive/MyDrive/MTS_NLP_models/hinglish_summarizer_50k'
out = '/content/drive/MyDrive/MTS_NLP_models/hinglish_summarizer_50k.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(src):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, src))
print('saved', out)
```

Then in Google Drive (web):

1. Right-click `hinglish_summarizer_50k.zip` → **Share** → **General access: Anyone with the link** (Viewer).
2. Copy the link — it looks like
   `https://drive.google.com/file/d/1AbC...XyZ/view?usp=sharing`
   The long id in the middle (`1AbC...XyZ`) is your `DRIVE_FILE_ID`.

> Already have a local trained model instead? Skip the zip — the API also
> works if you just place the model folder at `./models/hinglish_summarizer_50k`.
> And if your zip contains the model inside a top-level folder rather than
> at the root, that's fine too — the API handles both layouts.

## Step 2 — Run the FastAPI backend

```bash
cd mts_nlp_app
pip install -r requirements.txt

export DRIVE_FILE_ID=<paste-your-file-id>     # Windows PowerShell: $env:DRIVE_FILE_ID="..."
uvicorn api:app --port 8000
```

First start downloads the model from Drive (~1.2 GB, takes a few minutes),
then loads it. Check it's alive:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true,"model_path":"models/hinglish_summarizer_50k","model_host":"google-drive","device":"cpu"}
```

Interactive API docs: http://127.0.0.1:8000/docs

Test the endpoint directly:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/process-chat \
  -H "Content-Type: application/json" \
  -d '{"conversation": "Amit: Bhai, kal final client meeting hai. Sab ready hai kya?\nRahul: Database connect nahi ho raha. Main check karta hu."}'
# {"summary":"...","truncated":false}
```

Endpoint contract (same as the original `main.py`, plus the flag):

- `POST /api/v1/process-chat` — body `{"conversation": "<hinglish text>"}`
- Response `{"summary": "<english summary>", "truncated": <bool>}`
  (`truncated: true` when the input exceeded the model's 256-token limit)
- Empty conversation → **400** (this is fixed here; the old `main.py`
  returned 500 by mistake)

## Step 3 — Run the Streamlit frontend

In a second terminal (API still running):

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501, paste a Hinglish conversation, hit
**Generate Summary**. If the API runs somewhere else, set its URL in the
sidebar or with `export API_URL=http://<host>:8000`.

## Notes

- **First request is slow** — model download + load happen at API startup,
  and the first summary on CPU can take several seconds. A GPU (if PyTorch
  sees one) is used automatically.
- **Free Drive limits:** "Anyone with the link" files have download quotas;
  fine for personal/demo use. For public deployment, Hugging Face Hub is a
  better model host — the API only needs `AutoModelForSeq2SeqLM.from_pretrained`
  pointed at the hub repo instead of the Drive download in `api.py`.
- **Security:** with `ALLOWED_ORIGINS=*` (default, for local use) any
  website can call your API. If you deploy publicly, set
  `ALLOWED_ORIGINS` to your frontend's origin.
- Nothing here changes the model or generation settings: input 256 tokens,
  beam search 4, max 100 new tokens, repetition penalty 1.2 — identical to
  training and the original `main.py`.
