from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# 1. Define Request and Response Schemas
class TranslationRequest(BaseModel):
    conversation: str


class TranslationResponse(BaseModel):
    summary: str


# 2. Initialize the FastAPI app
app = FastAPI(title="Hinglish AI Summarizer")

# Enable CORS for browser security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the End-to-End mT5 Model (No DistilBART needed!)
print("Loading Custom mT5 Model...")
model_path = "./models/hinglish_summarizer_50k"
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

# 4. Dedicated Frontend UI
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Hinglish AI Summarizer</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 40px auto; background-color: #f4f4f9; padding: 20px; }
        h1 { color: #2c3e50; text-align: center; font-weight: 600; }
        textarea { width: 100%; height: 180px; padding: 15px; border: 2px solid #e0e0e0; border-radius: 10px; font-size: 16px; margin-bottom: 20px; box-sizing: border-box; resize: vertical; transition: border-color 0.3s; }
        textarea:focus { border-color: #3498db; outline: none; }
        button { background-color: #3498db; color: white; padding: 14px 24px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; width: 100%; font-weight: bold; transition: background-color 0.3s; }
        button:hover { background-color: #2980b9; }
        .output-box { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-top: 25px; display: none; border-left: 5px solid #27ae60; }
        h3 { margin-top: 0; color: #27ae60; font-size: 18px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        pre { white-space: pre-wrap; word-wrap: break-word; font-family: inherit; color: #333; font-size: 17px; line-height: 1.5; font-weight: 500;}
        .loader { display: none; text-align: center; margin-top: 20px; color: #7f8c8d; font-style: italic; font-size: 15px; }
    </style>
</head>
<body>
    <h1>Hinglish AI Summarizer</h1>
    <textarea id="inputText" placeholder="Paste your Hinglish conversation here...
Example:
Amit: Bhai, kal dopehar apna final client meeting hai. Sab ready hai kya?
Rahul: Database connect nahi ho raha. Main check karta hu."></textarea>

    <button onclick="processText()">Generate Summary</button>
    <div class="loader" id="loader">Processing text via local GPU... Please wait.</div>

    <div class="output-box" id="outputBox">
        <h3>Final English Summary</h3>
        <pre id="summaryOutput"></pre>
    </div>

    <script>
        async function processText() {
            const text = document.getElementById('inputText').value;
            if (!text.trim()) return alert("Please enter a conversation first!");

            document.getElementById('loader').style.display = 'block';
            document.getElementById('outputBox').style.display = 'none';

            try {
                const response = await fetch('/api/v1/process-chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ conversation: text })
                });

                const data = await response.json();

                if (response.ok) {
                    document.getElementById('summaryOutput').textContent = data.summary;
                    document.getElementById('outputBox').style.display = 'block';
                } else {
                    alert("API Error: " + (data.detail || "Invalid input data"));
                }
            } catch (error) {
                alert("Failed to connect to the server. Is Uvicorn running?");
            } finally {
                document.getElementById('loader').style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return html_content


# 5. Define the REST Endpoint (End-to-End matching Colab)
@app.post("/api/v1/process-chat", response_model=TranslationResponse)
async def process_chat(request: TranslationRequest):
    try:
        if not request.conversation.strip():
            raise HTTPException(status_code=400, detail="Conversation text cannot be empty.")

        # Replicating the exact Colab generation logic
        inputs = tokenizer(
            request.conversation.strip(),
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(device)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=4,
                repetition_penalty=1.2,
                early_stopping=True
            )

        final_summary = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        return TranslationResponse(summary=final_summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
