# 🛒 Return Policy Assistant

An AI-powered chatbot that reads any return policy PDF and answers your questions in plain English — with source citations from the actual policy document. No hallucinations.

> Ask: *"I got a broken phone. Can I return it?"*
> Get: *"Yes — damaged items can be returned within 10 days. Amazon covers return shipping."*

Built in 2 hours as a weekend project. Same pattern works for **any boring PDF** — credit card terms, insurance policies, legal contracts, employee handbooks.

---

## 🧠 How It Works (RAG Pipeline)

This project uses **RAG (Retrieval-Augmented Generation)** — the same pattern powering most production AI products today.

```
┌─────────────────────────────────────────────────────────┐
│  INGESTION (run once per PDF)                           │
│                                                         │
│  PDF → Extract text → Chunk → Embed → Store in ChromaDB │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  QUERYING (every question)                              │
│                                                         │
│  Question → Embed → Search top-3 chunks →               │
│  Send chunks + question to DeepSeek → Grounded answer   │
└─────────────────────────────────────────────────────────┘
```

**Why this matters:** The LLM never sees the whole PDF. It only sees the 3-5 most relevant chunks. That's what keeps answers accurate and grounded in the actual document — not the model's training data.

---

## 🚀 Setup

### Prerequisites

- Python 3.9+ ([download](https://www.python.org/downloads/))
- A DeepSeek API key ([get one here](https://platform.deepseek.com/)) — free tier available

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/refund-assistant.git
cd refund-assistant
```

### 2. Create a virtual environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a file called `.env` in the project root:

```
DEEPSEEK_API_KEY=sk-your-actual-key-here
```

Get your key from [platform.deepseek.com](https://platform.deepseek.com/).

### 5. Add a return policy PDF

Drop any return policy PDF into the `policies/` folder. Name it `amazon_return_policy.pdf` (or update `PDF_PATH` in `ingest.py`).

### 6. Ingest the PDF (run once)

```bash
python ingest.py
```

This will:
- Extract text from the PDF
- Split it into ~500-character chunks
- Convert each chunk into a vector embedding (downloads an 80MB model on first run)
- Store everything in a local ChromaDB

### 7. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Ask anything about the policy.

---

## 📁 Project Structure

```
refund-assistant/
├── .env                 # Your DeepSeek API key (never commit this)
├── .gitignore           # Files to exclude from Git
├── requirements.txt     # Python dependencies
├── ingest.py            # Reads PDF → chunks → embeds → stores in ChromaDB
├── app.py               # Streamlit chat UI
├── README.md            # This file
├── policies/            # Folder for your PDFs
│   └── amazon_return_policy.pdf
└── chroma_db/           # Vector DB (auto-generated, gitignored)
```

---

## 🛠️ Tech Stack

| Tool                                            | Purpose                        |
| ----------------------------------------------- | ------------------------------ |
| [Streamlit](https://streamlit.io/)              | Web UI in pure Python          |
| [sentence-transformers](https://www.sbert.net/) | Local embeddings (no API cost) |
| [ChromaDB](https://www.trychroma.com/)          | Vector database                |
| [pypdf](https://pypdf.readthedocs.io/)          | PDF text extraction            |
| [DeepSeek](https://platform.deepseek.com/)      | LLM for answer generation      |

---

## 🎛️ Tuning Knobs

In `ingest.py`:

| Variable        | Default | What it does                                                                       |
| --------------- | ------- | ---------------------------------------------------------------------------------- |
| `CHUNK_SIZE`    | 500     | Characters per chunk. Smaller = more precise retrieval, but might split sentences. |
| `CHUNK_OVERLAP` | 100     | Overlap between chunks so key ideas aren't cut in half.                            |

In `app.py`:

| Variable | Default | What it does                                                                      |
| -------- | ------- | --------------------------------------------------------------------------------- |
| `TOP_K`  | 3       | How many chunks to feed the LLM. Higher = more context, but slower + more tokens. |

**Try this experiment:** Set `CHUNK_OVERLAP = 0` and re-ingest. Watch how answers get worse — that's why overlap matters.

---

## 💡 Use Cases (Beyond Return Policies)

The exact same pipeline works for:

- 💳 **Credit card terms** — "What's the late fee?"
- 🏥 **Insurance policies** — "Does my plan cover dental?"
- ✈️ **Travel booking policies** — "Can I cancel for free?"
- 📜 **Legal contracts** — "What's the termination clause?"
- 📚 **Research papers** — "What method did they use?"
- 🏢 **Employee handbooks** — "How many sick days do I get?"

Just swap the PDF and re-run `ingest.py`.

---

## ⚠️ Notes

- **First run downloads ~80MB** (the embedding model). Subsequent runs are instant.
- **`torchvision` warning on startup** is harmless — it's from an unrelated model in `transformers`. Safe to ignore.
- **Never commit `.env`.** Your API key will be exposed publicly.

---

## 📄 License

MIT — do whatever you want with it.

---

## 🙌 Built With

Python · DeepSeek · ChromaDB · Streamlit · sentence-transformers

Built in one Sunday evening. If you build something with this, tag me!