"""
rag_evaluation.py
Evaluation framework for the RAG pipeline with representative questions and quality analysis.
"""
import pandas as pd
from typing import List, Dict, Any
from .rag_pipeline import retrieve_relevant_chunks, generate_answer
from .embedding import Embedder
from .vector_store import VectorStore
import time

# Representative questions for evaluation
EVALUATION_QUESTIONS = [
    # Product-specific questions
    "Why are people unhappy with Buy Now, Pay Later?",
    "What are the main issues with Credit Cards?",
    "What problems do customers face with Personal Loans?",
    "What complaints are common with Savings Accounts?",
    "What issues arise with Money Transfers?",
    
    # Cross-product analysis
    "What are the most common customer service issues across all products?",
    "What billing and payment problems affect multiple financial products?",
    "How do fraud and security concerns vary across different products?",
    
    # Business intelligence questions
    "What emerging trends should product managers be aware of?",
    "What are the most urgent compliance issues to address?"
]

def evaluate_rag_pipeline(questions: List[str] = None) -> pd.DataFrame:
    """
    Evaluate the RAG pipeline with a set of representative questions.
    
    Args:
        questions: List of questions to evaluate. If None, uses default questions.
        
    Returns:
        DataFrame with evaluation results
    """
    if questions is None:
        questions = EVALUATION_QUESTIONS
    
    print("🚀 Starting RAG Pipeline Evaluation...")
    print(f"📊 Evaluating {len(questions)} questions...")
    
    # Load vector store and embedder
    print("📚 Loading vector store and embedder...")
    store = VectorStore.load("vector_store/faiss_index.idx", "vector_store/metadata.pkl")
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32)
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n🔍 Question {i}/{len(questions)}: {question}")
        
        try:
            # Time the retrieval and generation
            start_time = time.time()
            
            # Retrieve relevant chunks
            top_chunks = retrieve_relevant_chunks(question, embedder, store, k=5)
            
            # Generate answer
            answer = generate_answer(question, top_chunks)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Extract source information
            sources = []
            for chunk in top_chunks[:2]:  # Show top 2 sources
                source_info = f"Product: {chunk.get('product', 'N/A')} | ID: {chunk.get('complaint_id', 'N/A')}"
                sources.append(f"{chunk['chunk'][:150]}... [{source_info}]")
            
            # Basic quality assessment
            quality_score = assess_response_quality(question, answer, top_chunks)
            
            results.append({
                'Question': question,
                'Generated_Answer': answer,
                'Retrieved_Sources': '\n\n'.join(sources),
                'Response_Time_Seconds': round(response_time, 2),
                'Quality_Score': quality_score,
                'Num_Sources': len(top_chunks),
                'Answer_Length': len(answer)
            })
            
            print(f"✅ Completed in {response_time:.2f}s | Quality: {quality_score}/5")
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            results.append({
                'Question': question,
                'Generated_Answer': f"Error: {str(e)}",
                'Retrieved_Sources': '',
                'Response_Time_Seconds': 0,
                'Quality_Score': 1,
                'Num_Sources': 0,
                'Answer_Length': 0
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add analysis
    df['Comments_Analysis'] = df.apply(lambda row: analyze_response(row), axis=1)
    
    print(f"\n🎉 Evaluation completed! Processed {len(results)} questions.")
    print(f"📈 Average Quality Score: {df['Quality_Score'].mean():.2f}/5")
    print(f"⏱️ Average Response Time: {df['Response_Time_Seconds'].mean():.2f}s")
    
    return df

def assess_response_quality(question: str, answer: str, sources: List[Dict[str, Any]]) -> int:
    """
    Assess the quality of a response on a scale of 1-5.
    
    Args:
        question: The original question
        answer: The generated answer
        sources: Retrieved source chunks
        
    Returns:
        Quality score (1-5)
    """
    score = 3  # Start with neutral score
    
    # Check if answer is complete
    if len(answer) < 50:
        score -= 1
    elif len(answer) > 200:
        score += 1
    
    # Check if answer addresses the question
    if any(word in answer.lower() for word in question.lower().split()):
        score += 1
    
    # Check if sources are relevant
    if sources and len(sources) >= 3:
        score += 1
    
    # Check for professional tone
    if any(word in answer.lower() for word in ['analysis', 'issue', 'problem', 'customer', 'financial']):
        score += 1
    
    # Penalize errors
    if 'error' in answer.lower():
        score -= 2
    
    return max(1, min(5, score))

def analyze_response(row: pd.Series) -> str:
    """
    Provide detailed analysis of a response.
    
    Args:
        row: DataFrame row with response data
        
    Returns:
        Analysis string
    """
    analysis = []
    
    # Quality assessment
    if row['Quality_Score'] >= 4:
        analysis.append("High quality response with good insights")
    elif row['Quality_Score'] >= 3:
        analysis.append("Acceptable response with room for improvement")
    else:
        analysis.append("Low quality response needs optimization")
    
    # Response time analysis
    if row['Response_Time_Seconds'] < 5:
        analysis.append("Fast response time")
    elif row['Response_Time_Seconds'] < 15:
        analysis.append("Moderate response time")
    else:
        analysis.append("Slow response time - consider optimization")
    
    # Source analysis
    if row['Num_Sources'] >= 4:
        analysis.append("Good source coverage")
    elif row['Num_Sources'] >= 2:
        analysis.append("Adequate source coverage")
    else:
        analysis.append("Limited source coverage")
    
    # Answer completeness
    if row['Answer_Length'] > 300:
        analysis.append("Comprehensive answer")
    elif row['Answer_Length'] > 100:
        analysis.append("Sufficient answer length")
    else:
        analysis.append("Answer may be too brief")
    
    return " | ".join(analysis)

def save_evaluation_results(df: pd.DataFrame, filename: str = "rag_evaluation_results.csv"):
    """
    Save evaluation results to CSV and generate summary.
    
    Args:
        df: Evaluation results DataFrame
        filename: Output filename
    """
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"💾 Results saved to {filename}")
    
    # Generate summary
    summary = f"""
# RAG Pipeline Evaluation Summary

## Overall Performance
- **Total Questions Evaluated**: {len(df)}
- **Average Quality Score**: {df['Quality_Score'].mean():.2f}/5
- **Average Response Time**: {df['Response_Time_Seconds'].mean():.2f}s
- **Success Rate**: {(df['Quality_Score'] > 1).sum()}/{len(df)} ({(df['Quality_Score'] > 1).sum()/len(df)*100:.1f}%)

## Quality Distribution
- **High Quality (4-5)**: {(df['Quality_Score'] >= 4).sum()} questions
- **Medium Quality (3)**: {(df['Quality_Score'] == 3).sum()} questions  
- **Low Quality (1-2)**: {(df['Quality_Score'] <= 2).sum()} questions

## Performance Insights
- **Fastest Response**: {df['Response_Time_Seconds'].min():.2f}s
- **Slowest Response**: {df['Response_Time_Seconds'].max():.2f}s
- **Average Answer Length**: {df['Answer_Length'].mean():.0f} characters

## Recommendations
1. **Optimize Response Quality**: Focus on questions with scores < 3
2. **Improve Response Time**: Investigate slow responses
3. **Enhance Source Retrieval**: Ensure adequate context coverage
4. **Refine Prompt Engineering**: Improve answer relevance and completeness
"""
    
    with open("evaluation_summary.md", "w") as f:
        f.write(summary)
    
    print("📝 Summary saved to evaluation_summary.md")
    return summary

if __name__ == "__main__":
    # Run evaluation
    results_df = evaluate_rag_pipeline()
    
    # Save results
    save_evaluation_results(results_df)
    
    # Display top results
    print("\n🏆 Top 3 Responses by Quality Score:")
    top_results = results_df.nlargest(3, 'Quality_Score')
    for _, row in top_results.iterrows():
        print(f"\nQuestion: {row['Question']}")
        print(f"Quality: {row['Quality_Score']}/5")
        print(f"Answer: {row['Generated_Answer'][:200]}...") 