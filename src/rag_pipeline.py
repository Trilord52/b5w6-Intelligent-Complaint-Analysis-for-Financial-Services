"""
rag_pipeline.py
RAG pipeline: retrieves relevant complaint chunks and generates answers using Hugging Face Inference API.
"""
import os
from typing import List, Dict, Any
from .embedding import Embedder
from .vector_store import VectorStore
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

# --- Model selection ---
# Change this variable to use a different Hugging Face model (see README for options)
HF_MODEL = "mistralai/Magistral-Small-2506"

# --- Hugging Face API setup ---
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_API_TOKEN,
)

# --- Enhanced Prompt template for high-quality responses ---
PROMPT_TEMPLATE = """You are a senior financial analyst at CrediTrust Financial, specializing in customer complaint analysis and business intelligence. Your role is to provide actionable insights to product managers, support teams, and compliance officers.

TASK: Analyze the provided customer complaint excerpts and answer the user's question with professional, data-driven insights.

RESPONSE REQUIREMENTS:
1. **Structure**: Provide a clear, well-organized response with bullet points or numbered lists when appropriate
2. **Length**: Aim for 200-500 words to ensure comprehensive coverage
3. **Tone**: Professional, analytical, and business-focused
4. **Content**: Include specific examples from the complaints, identify patterns, and suggest actionable insights
5. **Format**: Use markdown formatting for better readability

ANALYSIS FRAMEWORK:
- **Issue Identification**: What specific problems are customers experiencing?
- **Pattern Recognition**: Are there common themes or recurring issues?
- **Impact Assessment**: How do these issues affect customer satisfaction and business operations?
- **Recommendations**: What actions should teams take to address these concerns?

CONTEXT (Customer Complaint Excerpts):
{context}

USER QUESTION: {question}

RESPONSE:"""

def generate_answer(question: str, retrieved_chunks: List[Dict[str, Any]], max_new_tokens: int = 1024, temperature: float = 0.3) -> str:
    """
    Generate a comprehensive answer using the RAG pipeline with enhanced prompt engineering.
    
    Args:
        question: User's question about customer complaints
        retrieved_chunks: List of relevant complaint chunks
        max_new_tokens: Maximum tokens for response generation
        temperature: Creativity level (lower = more focused)
        
    Returns:
        Formatted, professional response with actionable insights
    """
    if not retrieved_chunks:
        return "I don't have enough information to provide a comprehensive answer. Please try rephrasing your question or ask about a different aspect of customer complaints."
    
    # Format context with better structure
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        product = chunk.get('product', 'Unknown Product')
        complaint_id = chunk.get('complaint_id', 'Unknown ID')
        context_parts.append(f"**Complaint {i}** (Product: {product}, ID: {complaint_id}):\n{chunk['chunk']}")
    
    context = "\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    try:
        # Use chat completions with simplified parameters for better compatibility
        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_new_tokens
        )
        
        # Enhanced response validation
        if completion and hasattr(completion, 'choices') and completion.choices:
            response = completion.choices[0].message.content
            if response and len(response.strip()) > 50:
                return response.strip()
            else:
                return "I apologize, but I couldn't generate a comprehensive response. Please try rephrasing your question or ask about a different aspect of customer complaints."
        else:
            return "I encountered an issue generating the response. Please try again or contact support if the problem persists."
            
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"I apologize, but I encountered a technical issue while processing your question. Please try again in a moment. If the problem persists, please contact the technical team."

def retrieve_relevant_chunks(question: str, embedder: Embedder, store: VectorStore, k: int = 7) -> List[Dict[str, Any]]:
    """
    Retrieve relevant complaint chunks using semantic search with enhanced coverage.
    
    Args:
        question: User's question
        embedder: Embedding model instance
        store: Vector store instance
        k: Number of chunks to retrieve (increased for better coverage)
        
    Returns:
        List of relevant complaint chunks with metadata
    """
    try:
        query_emb = embedder.embed_query(question)
        results, scores = store.similarity_search(query_emb, k=k)
        
        # Filter out very low similarity scores
        filtered_results = []
        for result, score in zip(results, scores):
            if score > 0.3:  # Minimum similarity threshold
                filtered_results.append(result)
        
        # Ensure we have at least 3 results for good coverage
        if len(filtered_results) < 3:
            # Get more results if we don't have enough after filtering
            more_results, _ = store.similarity_search(query_emb, k=k+3)
            for result in more_results:
                if result not in filtered_results and len(filtered_results) < 5:
                    filtered_results.append(result)
        
        return filtered_results[:5]  # Return top 5 for optimal context
        
    except Exception as e:
        print(f"Error in retrieval: {e}")
        return []

# Example usage (for evaluation or integration with UI):
if __name__ == "__main__":
    # Load vector store and embedder
    store = VectorStore.load("vector_store/faiss_index.idx", "vector_store/metadata.pkl")
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32)
    
    # Example question
    question = "Why are people unhappy with Buy Now, Pay Later?"
    top_chunks = retrieve_relevant_chunks(question, embedder, store, k=5)
    answer = generate_answer(question, top_chunks)
    print("Question:", question)
    print("\nAnswer:", answer)
    print("\nSources:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"[{i}] {chunk['chunk'][:200]}...") 