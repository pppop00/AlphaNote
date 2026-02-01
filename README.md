# AlphaNote

A personal RAG-powered investment thesis journal with local LLM integration.

## Features

- **Chat with your Notes** - Query historical investment theses using semantic search
- **Log New Thesis** - Record asset/ticker, market stance (Bullish/Neutral/Bearish), and reasoning
- **Accountability** - Each entry requires an author name
- **Recent Intelligence** - Preview your 6 most recent entries
- **Incremental RAG** - New entries are indexed instantly without full rebuild
- **Local LLM** - Private AI analysis via Ollama (llama3.2)
- **Persistent Storage** - Markdown files + ChromaDB vector database

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install & start Ollama (required for chat)
# Download from: https://ollama.ai
ollama pull llama3.2
ollama serve

# 3. Run AlphaNote
python app.py
```

Open **http://127.0.0.1:7860**

## File Structure

```
personal-RAG/
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── knowledge-base/         # Markdown storage
│   └── entries_001.md      # Investment thesis journal
└── vector_db/              # ChromaDB (auto-generated)
```

## Entry Format

Entries are stored in `knowledge-base/entries_XXX.md`:

```markdown
---
## GOLD | Bullish | 2026-01-31 14:30 | Oliver

Gold remains attractive as a hedge against uncertainty...
```

When a file exceeds 500KB, a new file (`entries_002.md`, etc.) is created automatically.

## Configuration

Edit `app.py` constants:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FILE_SIZE_KB` | 500 | File size limit before rotation |
| `USE_LOCAL_LLM` | True | Use Ollama (True) or OpenAI (False) |
| `LOCAL_MODEL` | llama3.2 | Ollama model name |
| `OPENAI_MODEL` | gpt-4o-mini | OpenAI model (if USE_LOCAL_LLM=False) |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |

## Performance Optimizations

- **Incremental Updates**: New entries are added to vector DB without rebuilding
- **Persistent Embeddings**: Model loaded once at startup
- **Disk-based VectorDB**: ChromaDB persists between restarts

## Requirements

- Python 3.10+
- Ollama running locally (`ollama serve`)
- ~2GB disk space for llama3.2 model
