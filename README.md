# Intelligent Complaint Analysis for Financial Services (CrediTrust)

A RAG-powered chatbot system for analyzing CFPB (Consumer Financial Protection Bureau) complaint data for CrediTrust Financial. This project implements intelligent complaint analysis using advanced NLP techniques and vector databases.

---

## 🎯 Project Overview & Business Value

TrustVoice Analytics enables CrediTrust teams to quickly extract actionable insights from thousands of raw, unstructured customer complaint narratives. By leveraging state-of-the-art NLP and retrieval-augmented generation (RAG), the system empowers product, support, and compliance teams to:
- Analyze and categorize CFPB consumer complaints
- Provide instant, evidence-backed answers to user queries
- Detect trends and issues proactively, reducing time-to-insight from days to minutes
- Transform raw complaint data into a user-friendly, internal tool

**This system provides intelligent analysis of financial complaints through:**
- **Interactive EDA and Preprocessing**: Comprehensive data exploration and cleaning
- **Vector Database Integration**: Efficient text chunking, embedding, and indexing
- **RAG Pipeline**: Advanced retrieval-augmented generation for complaint analysis
- **Interactive Chat Interface**: User-friendly web interface for complaint queries

---

## 📁 Project Structure

```
b5w6-Intelligent-Complaint-Analysis-for-Financial-Services/
├── data/                          # Raw and processed data files
│   └── complaints.csv             # CFPB complaints dataset (5.6GB)
│   └── complaints_processed.csv   # Cleaned and filtered data
├── notebooks/                     # Jupyter notebooks (EDA, preprocessing)
│   └── eda_preprocessing.ipynb
├── reports/                       # Generated reports and visualizations
├── src/                           # Modular source code (object-oriented, clear separation)
│   └── chunking.py                # Text chunking logic
│   └── embedding.py               # Embedding generation
│   └── vector_store.py            # FAISS vector store logic
│   └── chunk_embed_index.py       # Main pipeline script
├── vector_store/                  # Vector database storage
│   └── faiss_index.idx            # FAISS index (Task 2 output)
│   └── metadata.pkl               # Metadata for indexed chunks
├── app.py                         # (For future: main app or UI entry point)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

**This structure ensures:**
- Clear separation of data, code, notebooks, and outputs
- Systematic, descriptive file naming
- Easy navigation and maintainability

---

## Code Quality & Best Practices
- **Object-Oriented & Modular:** Each major function (chunking, embedding, indexing) is implemented in its own module, with clear interfaces and documentation.
- **Documentation:** Inline comments and docstrings explain the purpose and logic of each component.
- **Commit History:** Changes are made incrementally with descriptive commit messages, supporting reproducibility and review.
- **Scalability:** Batch processing and checkpointing enable robust handling of large datasets.
- **Extensibility:** The codebase is designed for easy addition of new features (e.g., RAG pipeline, chat interface).

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Data Preparation
- Place your raw CFPB complaints dataset as `data/complaints.csv`.
- After preprocessing, the cleaned data will be saved as `data/complaints_processed.csv`.

### CFPB Dataset Columns
The project uses the CFPB complaints dataset (`data/complaints.csv`) with the following columns:
- `complaint_id`: Unique identifier
- `date_received`: Date complaint was received
- `product`: Financial product type
- `sub_product`: Sub-category of product
- `issue`: Main complaint issue
- `sub_issue`: Sub-category of issue
- `consumer_complaint_narrative`: Detailed complaint description
- `company_public_response`: Company's public response
- `company`: Company name
- `state`: State where complaint originated
- `zip_code`: ZIP code
- `tags`: Additional tags
- `consumer_consent_provided`: Consent status
- `submitted_via`: Submission method
- `date_sent_to_company`: Date sent to company
- `company_response_to_consumer`: Company's response
- `timely_response`: Whether response was timely
- `consumer_disputed`: Whether consumer disputed the response

---

## Running the Project

### Task 1: EDA and Preprocessing
- Run the interactive notebook:
```bash
jupyter notebook notebooks/eda_preprocessing.ipynb
```
- The notebook provides:
  - Comprehensive data exploration
  - Data quality assessment
  - Text preprocessing and cleaning
  - Statistical analysis and visualizations
  - Data filtering and preparation for downstream tasks

### Task 2: Chunking, Embedding, and Indexing
```bash
python src/chunk_embed_index.py
```
- The script will process the data in batches, generate embeddings, and build the FAISS index.
- If interrupted, simply rerun the script; it will resume from the last completed batch.

**Outputs:**
- FAISS index: `vector_store/faiss_index.idx`
- Metadata: `vector_store/metadata.pkl`

---

## Tasks

### Task 1: EDA & Preprocessing ✅
- Performed exploratory data analysis on the CFPB dataset.
- Filtered for five product types.
- Cleaned and standardized complaint narratives.
- Saved the cleaned dataset to `data/complaints_processed.csv`.

### Task 2: Chunking, Embedding, and Indexing ✅
- Split cleaned complaint narratives into overlapping text chunks.
- Generated semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2` (with GPU acceleration and batching).
- Indexed embeddings and metadata in a FAISS vector store for efficient semantic search.
- Implemented batch processing and checkpointing for scalability and robustness.
- Outputs: `vector_store/faiss_index.idx` and `vector_store/metadata.pkl`.

### Task 3: RAG Pipeline and Evaluation (Coming Soon)
- Build a retriever function and prompt template for the LLM to generate context-based answers.
- Evaluate using a set of representative questions.

### Task 4: Interactive Chat Interface (Coming Soon)
- Create an interactive chat interface using Gradio or Streamlit, ensuring users can see source texts and clear previous chats.

---

## 🛠️ Technical Stack

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development environment
- **Sentence-Transformers**: Embedding generation
- **FAISS**: Vector database for semantic search
- **Streamlit/Gradio**: Web interface (planned)

---

## 📈 Performance Metrics

The system will be evaluated on:
- **Retrieval Accuracy**: Precision and recall of relevant complaints
- **Response Quality**: Relevance and helpfulness of generated responses
- **Processing Speed**: Time efficiency for real-time analysis
- **User Satisfaction**: Interface usability and response quality

---

## 📊 Features

- **Modular, object-oriented code** for each pipeline stage
- **Clear directory structure** and systematic file naming
- **Batch processing and checkpointing** for large-scale data
- **GPU acceleration** for fast embedding generation

---

## Next Steps

- **Task 3:** Build the RAG pipeline (retriever, prompt template, LLM integration, evaluation).
- **Task 4:** Develop an interactive chat interface (Gradio/Streamlit) for end users.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is developed for educational purposes as part of the 10 Academy KAIM 5 program.

---

## 👥 Team

- **CrediTrust Financial**: Project sponsor
- **10 Academy**: Educational institution
- **KAIM 5 Cohort**: Student developers

---

## 📞 Support

For questions or support, please refer to the project documentation or contact the development team.

---

## Acknowledgements

- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)

---

**Note**: This project is designed for educational purposes and demonstrates advanced NLP and machine learning techniques for financial data analysis.
