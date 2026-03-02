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
ollama pull nomic-embed-text
ollama serve

# 3. Run AlphaNote
python app.py
```

Open **http://127.0.0.1:7860**

If port `7860` is occupied, either:

```bash
ALPHANOTE_PORT=7861 python app.py
```

or stop old process first:

```bash
lsof -nP -iTCP:7860 -sTCP:LISTEN
kill <PID>
```

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
| `EMBEDDING_MODEL` | nomic-embed-text | Ollama embedding model |

## Startup Issue: Root Cause and Fix

### Root Cause

1. **Port conflict**: older AlphaNote processes were still listening on `7860`, so Gradio failed with:
   `Cannot find empty port in range...`.
2. **Environment drift with latest dependency versions**: eager imports in the original startup path (`langchain_huggingface`, `langchain_openai`, and heavy splitters/loaders) could block startup in this local Python/venv setup.

### Fix Implemented

1. **Made launch port configurable** via `ALPHANOTE_PORT`; if not set, app uses Gradio default behavior.
2. **Reduced startup-time heavy imports**:
   - switched embeddings to `OllamaEmbeddings` (`nomic-embed-text`)
   - replaced LangChain `ChatOpenAI` call path with direct Ollama HTTP call
   - replaced heavy text-splitter/doc-loader path with lightweight local chunking
3. **Deferred vector DB initialization** until first request, so UI can start quickly and reliably.

## Performance Optimizations

- **Incremental Updates**: New entries are added to vector DB without rebuilding
- **Persistent Embeddings**: Model loaded once at startup
- **Disk-based VectorDB**: ChromaDB persists between restarts

## Requirements

- Python 3.10+
- Ollama running locally (`ollama serve`)
- ~2GB disk space for llama3.2 model
