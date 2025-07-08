# Intelligent Complaint Analysis for Financial Services

A RAG-powered chatbot system for analyzing CFPB (Consumer Financial Protection Bureau) complaint data for CrediTrust Financial. This project implements intelligent complaint analysis using advanced NLP techniques and vector databases.

## 🎯 Project Overview

This system provides intelligent analysis of financial complaints through:
- **Interactive EDA and Preprocessing**: Comprehensive data exploration and cleaning
- **Vector Database Integration**: Efficient text chunking, embedding, and indexing
- **RAG Pipeline**: Advanced retrieval-augmented generation for complaint analysis
- **Interactive Chat Interface**: User-friendly web interface for complaint queries

## 📁 Project Structure

```
b5w6-Intelligent-Complaint-Analysis-for-Financial-Services/
├── data/                          # Data files
│   └── complaints.csv             # CFPB complaints dataset (5.6GB)
├── notebooks/                     # Jupyter notebooks
│   └── eda_preprocessing.ipynb    # Task 1: EDA and preprocessing
├── reports/                       # Generated reports and visualizations
├── vector_store/                  # Vector database storage
├── app.py                         # Main application entry point
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

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
- `complaint_id`: Complaint identifier

### 3. Running the Project

#### Task 1: EDA and Preprocessing
```bash
# Run the interactive notebook
jupyter notebook notebooks/eda_preprocessing.ipynb
```

The notebook provides:
- Comprehensive data exploration
- Data quality assessment
- Text preprocessing and cleaning
- Statistical analysis and visualizations
- Data filtering and preparation for downstream tasks

## 📊 Features

### Task 1: EDA and Preprocessing ✅
- **Data Exploration**: Comprehensive analysis of complaint patterns
- **Quality Assessment**: Identification of missing data and inconsistencies
- **Text Preprocessing**: Cleaning and normalization of complaint narratives
- **Statistical Analysis**: Distribution analysis and correlation studies
- **Visualization**: Interactive charts and graphs for insights

### Task 2: Text Chunking, Embedding, and Vector Store Indexing (Coming Soon)
- **Text Chunking**: Intelligent segmentation of complaint narratives
- **Embedding Generation**: High-quality vector representations
- **Vector Database**: Efficient storage and retrieval system

### Task 3: RAG Pipeline and Evaluation (Coming Soon)
- **Retrieval System**: Semantic search capabilities
- **Generation Pipeline**: Context-aware response generation
- **Evaluation Metrics**: Performance assessment and optimization

### Task 4: Interactive Chat Interface (Coming Soon)
- **Web Interface**: User-friendly chat application
- **Real-time Analysis**: Instant complaint insights
- **Professional UI**: Modern, responsive design

## 🛠️ Technical Stack

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development environment
- **Scikit-learn**: Machine learning utilities
- **NLTK/spaCy**: Natural language processing
- **ChromaDB**: Vector database (planned)
- **Streamlit**: Web interface (planned)

## 📈 Performance Metrics

The system will be evaluated on:
- **Retrieval Accuracy**: Precision and recall of relevant complaints
- **Response Quality**: Relevance and helpfulness of generated responses
- **Processing Speed**: Time efficiency for real-time analysis
- **User Satisfaction**: Interface usability and response quality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is developed for educational purposes as part of the 10 Academy KAIM 5 program.

## 👥 Team

- **CrediTrust Financial**: Project sponsor
- **10 Academy**: Educational institution
- **KAIM 5 Cohort**: Student developers

## 📞 Support

For questions or support, please refer to the project documentation or contact the development team.

---

**Note**: This project is designed for educational purposes and demonstrates advanced NLP and machine learning techniques for financial data analysis.
