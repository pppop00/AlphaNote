"""
AlphaNote - Personal Investment Thesis Journal
Optimized with Incremental RAG Updates
"""

import os
import glob
import requests
from datetime import datetime
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

# ================================
# Configuration
# ================================
KNOWLEDGE_BASE_DIR = "knowledge-base"
VECTOR_DB_DIR = "vector_db"
MAX_FILE_SIZE_KB = 500

EMBEDDING_MODEL = "nomic-embed-text"
USE_LOCAL_LLM = True
LOCAL_MODEL = "llama3.2"
OPENAI_MODEL = "gpt-4o-mini"

# ================================
# File Management
# ================================

def ensure_directories():
    Path(KNOWLEDGE_BASE_DIR).mkdir(parents=True, exist_ok=True)
    Path(VECTOR_DB_DIR).mkdir(parents=True, exist_ok=True)

def get_current_file_path() -> Path:
    ensure_directories()
    md_files = sorted(glob.glob(f"{KNOWLEDGE_BASE_DIR}/entries_*.md"))
    if not md_files:
        return Path(KNOWLEDGE_BASE_DIR) / "entries_001.md"
    current_file = Path(md_files[-1])
    if current_file.exists() and (current_file.stat().st_size / 1024) >= MAX_FILE_SIZE_KB:
        num = int(current_file.stem.split('_')[1]) + 1
        return Path(KNOWLEDGE_BASE_DIR) / f"entries_{num:03d}.md"
    return current_file

def add_entry_to_file(ticker: str, direction: str, thesis: str, author: str) -> str:
    """Write entry to file and return the entry text for RAG indexing."""
    if not ticker.strip() or not thesis.strip():
        return "⚠️ Please fill in Asset/Ticker and Investment Logic"
    if not author.strip():
        return "⚠️ Please enter your name for accountability"
    
    filepath = get_current_file_path()
    timestamp = datetime.now()
    
    entry_text = f"""
---

## {ticker.upper()} | {direction} | {timestamp.strftime('%Y-%m-%d %H:%M')} | {author.strip()}

{thesis}

"""
    
    with open(filepath, 'a', encoding='utf-8') as f:
        if not filepath.exists() or filepath.stat().st_size == 0:
            f.write(f"# Investment Thesis Journal\n\nCreated: {timestamp.strftime('%Y-%m-%d')}\n")
        f.write(entry_text)
    
    return entry_text  # Return for incremental RAG update

def format_recent_entries_html() -> str:
    md_files = sorted(glob.glob(f"{KNOWLEDGE_BASE_DIR}/entries_*.md"))
    if not md_files:
        return "<p style='color:#8e8e93;text-align:center;padding:20px;'>No entries yet.</p>"
    
    all_entries = []
    for filepath in md_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                parts = f.read().split('---')
                for part in parts:
                    if '## ' in part:
                        lines = part.strip().split('\n')
                        header = lines[0].replace('## ', '').split(' | ')
                        if len(header) >= 2:
                            content = " ".join(lines[1:]).strip()
                            all_entries.append({
                                'ticker': header[0],
                                'dir': header[1],
                                'date': header[2] if len(header) > 2 else "",
                                'author': header[3] if len(header) > 3 else "Unknown",
                                'preview': content[:80] + "..." if len(content) > 80 else content
                            })
        except:
            pass
    
    html = ""
    for e in reversed(all_entries[-6:]):
        color = "#27ae60" if "Bullish" in e['dir'] else "#e74c3c" if "Bearish" in e['dir'] else "#95a5a6"
        html += f"""
        <div style="background:white; border-left:4px solid {color}; padding:10px; margin-bottom:8px; border-radius:4px; box-shadow:0 1px 2px rgba(0,0,0,0.05);">
            <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <span style="font-weight:bold; color:#2c3e50;">{e['ticker']}</span>
                <span style="font-size:11px; color:#bdc3c7;">{e['date']}</span>
            </div>
            <div style="font-size:11px; color:#3498db; margin-bottom:4px;">by {e['author']}</div>
            <div style="font-size:12px; color:#7f8c8d; line-height:1.4;">{e['preview']}</div>
        </div>
        """
    return html

# ================================
# RAG Engine (Optimized)
# ================================

class ThesisRAG:
    def __init__(self):
        self.vectorstore = None
        self.llm = None
        self.is_ready = False
        
        # Load embedding model once at startup
        print("[RAG] Loading embedding model...")
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url="http://localhost:11434"
        )
        print("[RAG] Embedding model loaded")
        
        # Defer vector database loading/rebuild until first use to keep startup fast.
        print("[RAG] Vector database initialization deferred until first request")

    def _init_llm(self):
        """Initialize the LLM."""
        if USE_LOCAL_LLM:
            self.llm = {
                "provider": "ollama",
                "model": LOCAL_MODEL,
                "url": "http://localhost:11434/api/chat"
            }
        else:
            self.llm = None
        print(f"[RAG] LLM initialized: {LOCAL_MODEL if USE_LOCAL_LLM else 'disabled'}")

    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 100):
        """Simple local splitter to avoid heavy optional dependencies."""
        clean_text = text.strip()
        if not clean_text:
            return []

        chunks = []
        start = 0
        text_len = len(clean_text)

        while start < text_len:
            end = min(text_len, start + chunk_size)
            chunks.append(clean_text[start:end])
            if end >= text_len:
                break
            start = max(end - overlap, start + 1)

        return chunks

    def _full_rebuild(self):
        """Full rebuild of vector database (only when needed)."""
        md_files = glob.glob(f"{KNOWLEDGE_BASE_DIR}/entries_*.md")
        if not md_files:
            print("[RAG] No files found for rebuild")
            return False
        
        try:
            print("[RAG] Starting full rebuild...")

            texts = []
            metadatas = []
            for file_path in md_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                chunks = self._split_text(content)
                texts.extend(chunks)
                metadatas.extend([{"source": file_path}] * len(chunks))

            print(f"[RAG] Split into {len(texts)} chunks")
            if not texts:
                print("[RAG] No content found for rebuild")
                return False

            self.vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                persist_directory=VECTOR_DB_DIR
            )
            
            self._init_llm()
            self.is_ready = True
            print("[RAG] Full rebuild complete")
            return True
            
        except Exception as e:
            print(f"[RAG] Rebuild error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add_incremental_entry(self, entry_text: str) -> bool:
        """Add a single entry to the vector database (incremental update)."""
        if not self.is_ready:
            # First time - do full rebuild
            print("[RAG] Not ready, performing full rebuild first")
            return self._full_rebuild()
        
        try:
            print("[RAG] Adding incremental entry...")
            self.vectorstore.add_texts(
                [entry_text],
                metadatas=[{"source": "incremental", "timestamp": datetime.now().isoformat()}]
            )
            print("[RAG] Incremental entry added successfully")
            return True
        except Exception as e:
            print(f"[RAG] Incremental add error: {e}")
            return False

    def chat(self, message: str, history: list):
        """Chat with the RAG system."""
        if not message.strip():
            return "", history
        
        new_history = list(history)
        new_history.append({"role": "user", "content": message})
        
        # Auto-initialize if not ready
        if not self.is_ready:
            print("[RAG] Not ready, initializing...")
            if not self._full_rebuild():
                new_history.append({
                    "role": "assistant",
                    "content": "⚠️ Knowledge base is empty or Ollama is not running. Please add entries and ensure `ollama serve` is running."
                })
                return "", new_history
        
        try:
            # Search for relevant documents
            print(f"[RAG] Searching for: {message[:50]}...")
            docs = self.vectorstore.similarity_search(message, k=5)
            print(f"[RAG] Found {len(docs)} relevant chunks")
            
            if not docs:
                new_history.append({
                    "role": "assistant",
                    "content": "I couldn't find any relevant entries in your thesis journal."
                })
                return "", new_history
            
            # Build context
            context_parts = []
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"[Entry {i}]\n{doc.page_content}")
            context = "\n\n".join(context_parts)
            
            # Create messages
            system_prompt = """You are an investment analyst assistant helping review historical investment theses.

Your role:
- Answer based ONLY on the provided context from the user's thesis journal
- Reference specific tickers, dates, and authors when available
- Be concise and professional
- If the context doesn't contain relevant information, say so

Do NOT make up information not in the context."""

            user_prompt = f"""Relevant entries from my investment thesis journal:

{context}

---

Question: {message}

Provide a clear answer based on the entries above."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM
            print("[RAG] Calling LLM...")
            if not USE_LOCAL_LLM or not self.llm:
                raise RuntimeError("Only local Ollama chat is supported in this environment.")

            response = requests.post(
                self.llm["url"],
                json={
                    "model": self.llm["model"],
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=120
            )
            response.raise_for_status()
            response_json = response.json()
            response_text = (response_json.get("message") or {}).get("content", "").strip()
            print(f"[RAG] Response received ({len(response_text)} chars)")
            
            if not response_text:
                response_text = "⚠️ Empty response from Ollama."

            new_history.append({"role": "assistant", "content": response_text})
            return "", new_history
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            if "Connection" in error_msg or "refused" in error_msg:
                error_msg = "⚠️ Cannot connect to Ollama. Please ensure `ollama serve` is running."
            else:
                error_msg = f"⚠️ Error: {error_msg}"
            
            new_history.append({"role": "assistant", "content": error_msg})
            return "", new_history

# Initialize RAG engine at startup
print("=" * 50)
print("Starting AlphaNote...")
print("=" * 50)
rag_engine = ThesisRAG()

# ================================
# UI
# ================================

css = """
.gradio-container { background-color: #F8F9FA !important; font-family: 'Inter', sans-serif !important; }
.custom-card { background: white; padding: 20px; border-radius: 8px; border: 1px solid #E9ECEF; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
.sidebar-header { font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: #6C757D; font-weight: 600; margin-bottom: 12px; }
.action-btn { background-color: #2C3E50 !important; color: white !important; border: none !important; }
.action-btn:hover { background-color: #34495E !important; }
"""

def create_app():
    with gr.Blocks(css=css, title="AlphaNote", theme=gr.themes.Default(primary_hue="slate")) as app:
        
        # Header
        gr.HTML("""
        <div style="background:#2C3E50; padding:15px 30px; display:flex; justify-content:space-between; align-items:center; border-radius:8px 8px 0 0;">
            <div style="color:white; font-size:18px; font-weight:600;">AlphaNote</div>
        </div>
        """)

        with gr.Row():
            # LEFT COLUMN
            with gr.Column(scale=35):
                
                # Log New Thesis
                with gr.Column(elem_classes="custom-card"):
                    gr.HTML("<div class='sidebar-header'>Log New Thesis</div>")
                    ticker = gr.Textbox(label="Asset / Ticker", placeholder="e.g. VTI, GOLD, USD/JPY")
                    direction = gr.Radio(["Bullish", "Neutral", "Bearish"], value="Bullish", label="Market Stance")
                    thesis = gr.TextArea(label="Investment Logic", placeholder="Describe the macro drivers, valuation, or catalysts...", lines=4)
                    author = gr.Textbox(label="Your Name", placeholder="Who's accountable for this thesis?")
                    log_btn = gr.Button("Log Thesis", variant="primary", elem_classes="action-btn")
                    status_msg = gr.Markdown()

                gr.HTML("<div style='height:16px;'></div>")
                
                # Recent Entries
                with gr.Column(elem_classes="custom-card"):
                    gr.HTML("<div class='sidebar-header'>Recent Intelligence</div>")
                    history_display = gr.HTML(format_recent_entries_html())
                    refresh_btn = gr.Button("Refresh", size="sm")

            # RIGHT COLUMN - Chat
            with gr.Column(scale=65):
                with gr.Column(elem_classes="custom-card"):
                    gr.HTML("<div class='sidebar-header'>Chat with your Notes</div>")
                    
                    chatbot = gr.Chatbot(
                        height=480,
                        show_label=False,
                        avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=alpha&backgroundColor=2C3E50")
                    )
                    
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask about your investment theses (e.g., 'What was my Gold thesis?')",
                            container=False,
                            scale=8
                        )
                        send_btn = gr.Button("Send", scale=2, elem_classes="action-btn")
                    
                    with gr.Row():
                        gr.HTML("<div style='flex:1'></div>")
                        clear_btn = gr.Button("Clear Chat", size="sm")

        # Event Handlers
        def handle_log(t, d, th, auth):
            # 1. Write to file
            entry_text = add_entry_to_file(t, d, th, auth)
            
            # Check for errors
            if entry_text.startswith("⚠️"):
                return entry_text, format_recent_entries_html(), t, th, auth
            
            # 2. Incremental RAG update (no full rebuild!)
            rag_engine.add_incremental_entry(entry_text)
            
            # 3. Return success
            return f"✓ {t.upper()} logged by {auth}", format_recent_entries_html(), "", "", ""

        log_btn.click(
            handle_log,
            [ticker, direction, thesis, author],
            [status_msg, history_display, ticker, thesis, author]
        )
        
        refresh_btn.click(format_recent_entries_html, None, history_display)
        
        chat_input.submit(rag_engine.chat, [chat_input, chatbot], [chat_input, chatbot])
        send_btn.click(rag_engine.chat, [chat_input, chatbot], [chat_input, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, chat_input])
        
        # Load entries on page load
        app.load(format_recent_entries_html, outputs=[history_display])

    return app

if __name__ == "__main__":
    ensure_directories()
    app = create_app()
    launch_kwargs = {"server_name": "0.0.0.0"}
    if os.getenv("ALPHANOTE_PORT"):
        launch_kwargs["server_port"] = int(os.getenv("ALPHANOTE_PORT"))
    app.launch(**launch_kwargs)
