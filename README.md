# 🏦 TrustVoice Analytics: Intelligent Complaint Analysis for Financial Services

TrustVoice Analytics is an advanced Retrieval-Augmented Generation (RAG) chatbot designed to help CrediTrust Financial turn unstructured customer complaint data into actionable business insights. The system leverages state-of-the-art natural language processing, semantic search, and large language models to empower product, support, and compliance teams to quickly identify pain points, trends, and opportunities across five major financial products.

---

## 🚩 Business Objective

CrediTrust Financial operates across East Africa, offering Credit Cards, Personal Loans, Buy Now Pay Later (BNPL), Savings Accounts, and Money Transfers. The company receives thousands of customer complaints monthly. This project delivers an internal AI tool that enables teams to:
- Instantly analyze and synthesize complaint narratives
- Surface emerging issues and trends in minutes
- Make data-driven decisions without technical expertise

---

## 🧩 Solution Overview

TrustVoice Analytics is a modular, production-ready RAG system that:
- Cleans and preprocesses raw complaint data
- Chunks long narratives for optimal semantic search
- Generates vector embeddings and builds a FAISS vector store
- Retrieves the most relevant complaint excerpts for any user query
- Uses a powerful LLM to generate concise, context-grounded answers
- Provides an interactive Gradio web interface for seamless user interaction

---

## 🗂️ Repository Structure

```
├── data/                  # Cleaned and raw complaint data (.gitkeep for empty dir)
├── notebooks/             # EDA and preprocessing notebooks
├── src/                   # Core RAG pipeline modules
│   ├── chunk_embed_index.py   # Data cleaning, chunking, embedding, and indexing
│   ├── rag_pipeline.py        # RAG retriever, prompt engineering, and LLM integration
│   ├── rag_evaluation.py      # Evaluation framework and quality scoring
│   ├── embedding.py           # Embedding model utilities
│   ├── vector_store.py        # FAISS vector store operations and metadata handling
│   └── chunking.py            # Text chunking utilities
├── vector_store/          # Persisted FAISS index and metadata
├── reports/               # Evaluation summary and results
│   └── evaluation_summary.md
├── app.py                 # Gradio chat interface for end users
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API Key**
   - Create a `.env` file in your project root:
     ```
     HF_API_TOKEN=your_huggingface_token_here
     ```

3. **Run the Chatbot Interface**
   ```bash
   python app.py
   ```
   - Access at [http://localhost:7860](http://localhost:7860)

---

## 🧑‍💻 Project Workflow & Components

### 1. **Data Preprocessing & EDA**
- **What was done:**
  - Loaded CFPB complaint data and filtered for five key products.
  - Performed exploratory data analysis (EDA) to understand product distribution, narrative lengths, and missing data.
  - Cleaned narratives by lowercasing, removing boilerplate, and filtering out empty entries.
- **How to run:**
  - See `notebooks/eda_preprocessing.ipynb` for EDA and cleaning steps.
  - The script `src/chunk_embed_index.py` automates cleaning and prepares data for chunking and embedding.
- **Output:**
  - Cleaned data saved to `data/filtered_complaints.csv`.

### 2. **Text Chunking, Embedding & Vector Store**
- **What was done:**
  - Long complaint narratives are split into overlapping 100-word chunks for better semantic search.
  - Each chunk is embedded using the `sentence-transformers/all-MiniLM-L6-v2` model, chosen for its speed and semantic accuracy.
  - Embeddings and metadata (complaint ID, product) are stored in a FAISS vector database for fast similarity search.
- **How to run:**
  - Execute `python src/chunk_embed_index.py` to process, chunk, embed, and index the data.
- **Output:**
  - Vector store files in `vector_store/` (e.g., `faiss_index.idx`, `metadata.pkl`).

### 3. **RAG Core Logic: Retrieval, Prompting & Generation**
- **What was done:**
  - Developed a retriever that embeds user questions and fetches the top-5 most relevant complaint chunks from the vector store.
  - Designed a prompt template that instructs the LLM to act as a financial analyst, use only the provided context, and admit when information is missing.
  - Integrated a large language model (default: `mistralai/Magistral-Small-2506` via Hugging Face Inference API) to generate answers grounded in retrieved context.
- **How to run:**
  - Use `src/rag_pipeline.py` to test retrieval and answer generation.
  - The prompt and model can be customized in this script.
- **Output:**
  - Answers with cited sources, ready for display in the UI or evaluation.

### 4. **Evaluation Framework**
- **What was done:**
  - Created a robust evaluation script (`src/rag_evaluation.py`) to test the system on 10 representative business questions.
  - Each answer is scored for quality, completeness, and source attribution.
  - Results are summarized in Markdown and CSV for easy reporting.
- **How to run:**
  - Execute `python src/rag_evaluation.py`.
- **Output:**
  - Evaluation summary in `reports/evaluation_summary.md`.

### 5. **Interactive Chat Interface**
- **What was done:**
  - Built a Gradio web app (`app.py`) for internal users to ask questions and receive synthesized, evidence-backed answers.
  - The interface features a text input, submit and clear buttons, answer display, and source chunk display for transparency.
  - Designed for ease of use by non-technical staff.
- **How to run:**
  - Start with `python app.py` and open the provided local URL.
- **Features:**
  - Real-time answer generation
  - Source attribution for trust
  - Conversation history
  - Clear/reset functionality

---

## 📊 Evaluation & Results

- **Average Quality Score:** 4.5/5 (see `reports/evaluation_summary.md`)
- **Success Rate:** 100% (10/10 business questions answered)
- **Average Response Time:** 53.35s
- **Key Improvements:**
  - Eliminated format errors
  - Increased answer length and structure
  - Enhanced context utilization

---

## 🛠️ Troubleshooting & Tips

- **Model Unavailable?** Try a different model (see below) or check your API token.
- **Slow Responses?** Hosted LLMs may be slow; use a smaller or local model for speed.
- **Streaming:** Not enabled by default. For most use cases, full response display is sufficient.
- **API Errors:** Ensure `.env` is present and contains a valid `HF_API_TOKEN`.

---

## 🤖 Model Selection & Customization

- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (fast, accurate, open-source)
- **LLM:** `mistralai/Magistral-Small-2506` (default)
- **Alternative Models:**
  - `gpt2`, `distilgpt2`, `tiiuae/falcon-7b-instruct`, or any Hugging Face Inference API-compatible model
- **To change model:** Edit `HF_MODEL` in `src/rag_pipeline.py`

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.

---

## 📄 License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

---

## 📚 References

- [CFPB Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [Hugging Face Inference API](https://huggingface.co/docs/huggingface_hub/v0.21.4/en/inference)
- [Mistral AI Documentation](https://huggingface.co/mistralai/Magistral-Small-2506)
- [Gradio Documentation](https://www.gradio.app/docs)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangChain](https://github.com/langchain-ai/langchain)

---

## 🏆 Project Status

- ✅ **Data Preprocessing & EDA:** Complete
- ✅ **Chunking, Embedding & Vector Store:** Complete
- ✅ **RAG Core Logic & Evaluation:** Complete
- ✅ **Interactive Chat Interface:** Complete
