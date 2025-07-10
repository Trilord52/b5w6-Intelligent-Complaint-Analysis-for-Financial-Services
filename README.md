# 🏦 TrustVoice Analytics
## Intelligent Complaint Analysis for Financial Services

A Retrieval-Augmented Generation (RAG) powered chatbot that transforms CFPB complaint data into actionable insights for CrediTrust Financial's internal teams.

### 🎯 Business Objective

CrediTrust Financial serves 500,000+ users across East African markets with Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers. This AI tool empowers product managers, support teams, and compliance officers to quickly identify customer pain points and emerging trends.

**Key Performance Indicators:**
- Reduce trend identification time from days to minutes
- Enable non-technical teams to get insights without data analysts
- Shift from reactive to proactive problem identification

### 🏗️ Project Architecture

```
├── data/                          # Processed complaint data
├── src/                           # Core RAG pipeline components
│   ├── chunk_embed_index.py      # Task 2: Chunking, embedding, indexing
│   ├── rag_pipeline.py           # Task 3: RAG retriever and generator
│   ├── rag_evaluation.py         # Evaluation framework
│   ├── embedding.py              # Embedding utilities
│   ├── vector_store.py           # FAISS vector store operations
│   └── chunking.py               # Text chunking utilities
├── vector_store/                  # FAISS index and metadata
├── app.py                        # Task 4: Interactive Gradio interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API Key:**
   Create a `.env` file in your project root:
   ```
   HF_API_TOKEN=your_huggingface_token_here
   ```

3. **Run the Interactive Interface:**
   ```bash
   python app.py
   ```
   The interface will be available at `http://localhost:7860`

### 🔧 Technical Implementation

#### **Task 1: Data Preprocessing** ✅
- **Dataset**: CFPB complaint data with 5 product categories
- **Processing**: Text cleaning, filtering, standardization
- **Output**: `data/complaints_processed.csv`

#### **Task 2: Vector Database** ✅
- **Chunking**: 100-word chunks with 20-word overlap
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS with metadata storage
- **Output**: `vector_store/faiss_index.idx` and `vector_store/metadata.pkl`

#### **Task 3: RAG Pipeline** ✅
- **Retriever**: Semantic search with top-k retrieval (k=5)
- **Generator**: Mistral AI Magistral-Small-2506 via FeatherlessAI
- **Prompt Engineering**: Financial analyst assistant role
- **Evaluation**: 10 representative questions with quality scoring

#### **Task 4: Interactive Interface** ✅
- **Framework**: Gradio web interface
- **Features**: 
  - Text input for questions
  - Real-time answer generation
  - Source attribution display
  - Conversation history
  - Clear functionality

### 🤖 Model Selection

#### **Embedding Model**
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Reason**: Fast, lightweight, good semantic understanding
- **Performance**: Optimized for CPU/GPU with mixed precision

#### **Language Model**
- **Model**: `mistralai/Magistral-Small-2506`
- **Provider**: FeatherlessAI (Hugging Face Inference API)
- **Capabilities**: 
  - 24B parameters with reasoning capabilities
  - 128k context window (recommended: 40k)
  - Multilingual support
  - Apache 2.0 license

#### **Changing Models**
To use a different model, update the `HF_MODEL` variable in `src/rag_pipeline.py`:
```python
HF_MODEL = "your-model-name-here"
```

### 📊 Evaluation Results

**Overall Performance:**
- **Average Quality Score**: 3.80/5
- **Success Rate**: 100% (10/10 questions processed)
- **Average Response Time**: 56.29s
- **High Quality Responses**: 6/10 questions

**Quality Distribution:**
- High Quality (4-5): 6 questions
- Medium Quality (3): 0 questions
- Low Quality (1-2): 4 questions

**Top Performing Questions:**
1. "Why are people unhappy with Buy Now, Pay Later?" (5/5)
2. "What are the main issues with Credit Cards?" (5/5)
3. "What are the most common customer service issues?" (5/5)

### 💡 Usage Examples

#### **Product-Specific Analysis**
```
Q: Why are people unhappy with Buy Now, Pay Later?
A: Based on complaint analysis, main issues include late fees, 
   misleading offers, loss of trust, poor customer service, 
   and perceived exploitation of customers.
```

#### **Cross-Product Insights**
```
Q: What are the most common customer service issues?
A: Lack of responsiveness, inability to speak with live 
   representatives, ineffective problem-solving, and 
   overall poor service quality across all products.
```

#### **Business Intelligence**
```
Q: What emerging trends should product managers be aware of?
A: Customer expectations for transparency, regulatory 
   compliance risks, internal communication gaps, and 
   the need for proactive issue management.
```

### 🔍 API Configuration

#### **Hugging Face Inference Setup**
1. Create account at [Hugging Face](https://huggingface.co)
2. Generate API token in Settings → Access Tokens
3. Add token to `.env` file as `HF_API_TOKEN`

#### **Alternative Models**
If you encounter issues with the default model, try:
- `gpt2` (always available)
- `distilgpt2` (lightweight)
- `tiiuae/falcon-7b-instruct` (if available)
- Local models via transformers library

### 🛠️ Troubleshooting & Tips

- **Model Unavailable?** If the default LLM is not available via Hugging Face Inference, try switching to a different model (see 'Alternative Models' above) or check your API token permissions.
- **Slow Responses?** Hosted LLMs may be slow. For faster responses, consider running a local model or using a smaller model.
- **Streaming:** Token-by-token streaming is not enabled by default. For most business use cases, full response display is sufficient. If you wish to enable streaming, see Gradio documentation for guidance.
- **API Errors:** Ensure your `.env` file is present and contains a valid `HF_API_TOKEN`.

### 🛠️ Development

#### **Running Individual Components**
```bash
# Test RAG pipeline
python src/rag_pipeline.py

# Run evaluation
python src/rag_evaluation.py

# Rebuild vector store
python src/chunk_embed_index.py
```

#### **Custom Questions**
Add your own questions to `src/rag_evaluation.py`:
```python
CUSTOM_QUESTIONS = [
    "Your question here?",
    "Another question?",
]
```

### 📈 Performance Optimization

#### **Response Time**
- Average: 56.29s (acceptable for business use)
- Optimization: Consider local model deployment for faster responses

#### **Quality Improvements**
- Prompt engineering refinements
- Source retrieval optimization
- Model parameter tuning

### 🔒 Security & Privacy

- API keys stored in environment variables
- No sensitive data in code repository
- Local processing of complaint data
- Secure API communication

### 📝 Reporting

The system generates comprehensive reports:
- `rag_evaluation_results.csv`: Detailed evaluation data
- `evaluation_summary.md`: Performance summary
- Interactive interface for real-time analysis

### 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.

### 📄 License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.

### 📚 References

- [CFPB Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [Hugging Face Inference API](https://huggingface.co/docs/huggingface_hub/v0.21.4/en/inference)
- [Mistral AI Documentation](https://huggingface.co/mistralai/Magistral-Small-2506)
- [Gradio Documentation](https://www.gradio.app/docs)

### 🏆 Project Status

- ✅ **Task 1**: EDA and Preprocessing - Complete
- ✅ **Task 2**: Vector Database - Complete  
- ✅ **Task 3**: RAG Pipeline - Complete
- ✅ **Task 4**: Interactive Interface - Complete

**Ready for production use and further development!**

---

*Developed for 10 Academy KAIM 5 Week 6 Challenge | CrediTrust Financial*
