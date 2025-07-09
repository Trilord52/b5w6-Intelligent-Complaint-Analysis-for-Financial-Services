"""
rag_pipeline.py
RAG pipeline: retrieves relevant complaint chunks and generates answers using Hugging Face Inference API.
"""
import os
from typing import List, Dict, Any
from embedding import Embedder
from vector_store import VectorStore
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

# --- Prompt template ---
PROMPT_TEMPLATE = (
    "You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. "
    "Use the following retrieved complaint excerpts to formulate your answer. "
    "If the context doesn't contain the answer, state that you don't have enough information.\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

def generate_answer(question: str, retrieved_chunks: List[Dict[str, Any]], max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    context = "\n---\n".join(chunk["chunk"] for chunk in retrieved_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    try:
        # Use chat completions with Mistral AI's Magistral-Small-2506
        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=0.95
        )
        
        # Check if completion and choices exist
        if completion and hasattr(completion, 'choices') and completion.choices:
            response = completion.choices[0].message.content
            if response:
                return response.strip()
            else:
                return "No response generated from the model."
        else:
            return "Invalid response format from the model."
            
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error generating response: {str(e)}"

def retrieve_relevant_chunks(question: str, embedder: Embedder, store: VectorStore, k: int = 5) -> List[Dict[str, Any]]:
    query_emb = embedder.embed_query(question)
    results, _ = store.similarity_search(query_emb, k=k)
    return results

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