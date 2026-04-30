# 📄 Chat with your PDF

> A Retrieval-Augmented Generation (RAG) app that lets you have a grounded Q&A conversation with any PDF — powered by LangChain, FAISS, HuggingFace Embeddings, and Groq (Llama 3.1).

---

## 🚀 Features

- 📤 **PDF Upload & Indexing** — Upload any PDF and it gets automatically chunked and indexed
- 🔒 **Grounded Answers Only** — The model answers strictly from your document, no hallucinations
- 💬 **Multi-turn Chat** — Full conversational memory within the session
- 📌 **Source Citations** — Every answer shows the exact page numbers referenced
- 🔄 **Auto Context Reset** — Chat history and index reset automatically when a new PDF is uploaded

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | `sentence-transformers/paraphrase-MiniLM-L3-v2` |
| Vector Store | FAISS (in-memory) |
| LLM | Groq — `llama-3.1-8b-instant` |
| PDF Parsing | PyPDFLoader |

---

## 📁 Project Structure

```
chat-with-pdf/
├── app.py              # Main Streamlit application
├── .env                # Environment variables (API keys)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/chat-with-pdf.git
cd chat-with-pdf
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📦 requirements.txt

```
streamlit
langchain
langchain-community
langchain-groq
langchain-text-splitters
faiss-cpu
sentence-transformers
pypdf
python-dotenv
```

---

## 🧠 How It Works

```
PDF Upload
    ↓
Split into chunks (500 tokens, 50 overlap)
    ↓
Embed with HuggingFace sentence-transformers
    ↓
Store in FAISS vector index
    ↓
User asks a question
    ↓
Top relevant chunks retrieved
    ↓
Context + chat history sent to Llama 3.1 on Groq
    ↓
Answer generated strictly from document context
    ↓
Response shown with page citations
```

---

## 💬 Usage

1. Launch the app and upload a PDF using the file uploader
2. Wait for indexing to complete
3. Type your question in the chat input
4. Get answers with page citations from your PDF
5. Upload a new PDF anytime — session resets automatically

---

## ⚙️ Configuration

**Tune chunking** in `app.py`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # characters per chunk
    chunk_overlap=50  # overlap between chunks
)
```

**Swap the LLM model**:

```python
llm = ChatGroq(model_name="llama-3.1-70b-versatile")  # more powerful
llm = ChatGroq(model_name="mixtral-8x7b-32768")        # larger context
```

---

## ⚠️ Limitations

- Scanned / image-only PDFs are not supported (no OCR)
- Vector store is in-memory and resets on app restart
- Very large PDFs may slow indexing

---

## 📄 License

MIT License — feel free to use and modify.

---

> Made with ❤️ using LangChain, Streamlit, and Groq
