"""
app.py
Interactive chat interface for the RAG-powered complaint analysis system.
Uses Gradio to create a user-friendly web interface for non-technical users.
"""
import gradio as gr
import os
from typing import List, Dict, Any
from src.rag_pipeline import retrieve_relevant_chunks, generate_answer
from src.embedding import Embedder
from src.vector_store import VectorStore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables for loaded components
store = None
embedder = None

def load_components():
    """Load vector store and embedder components."""
    global store, embedder
    if store is None:
        print("📚 Loading vector store and embedder...")
        store = VectorStore.load("vector_store/faiss_index.idx", "vector_store/metadata.pkl")
        embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32)
        print("✅ Components loaded successfully!")

def format_sources(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved sources for display."""
    if not chunks:
        return "No sources found."
    
    formatted_sources = []
    for i, chunk in enumerate(chunks, 1):
        source_info = f"Product: {chunk.get('product', 'N/A')} | ID: {chunk.get('complaint_id', 'N/A')}"
        formatted_sources.append(f"**Source {i}:**\n{chunk['chunk'][:300]}...\n*{source_info}*\n")
    
    return "\n".join(formatted_sources)

def query_rag(question: str, history: List[List[str]]) -> tuple:
    """
    Process a question through the RAG pipeline.
    
    Args:
        question: User's question
        history: Chat history (not used in current implementation)
        
    Returns:
        Tuple of (answer, sources, updated_history)
    """
    if not question.strip():
        return "", "", history
    
    try:
        # Load components if not already loaded
        load_components()
        
        # Retrieve relevant chunks
        top_chunks = retrieve_relevant_chunks(question, embedder, store, k=5)
        
        # Generate answer
        answer = generate_answer(question, top_chunks)
        
        # Format sources
        sources = format_sources(top_chunks)
        
        # Update history
        history.append([question, answer])
        
        return answer, sources, history
        
    except Exception as e:
        error_msg = f"Error processing your question: {str(e)}"
        return error_msg, "No sources available due to error.", history

def clear_chat() -> tuple:
    """Clear the chat history."""
    return "", "", []

def create_interface():
    """Create the Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .source-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
    }
    .answer-box {
        background-color: #e8f5e8;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    """
    
    with gr.Blocks(css=css, title="TrustVoice Analytics - Complaint Analysis") as interface:
        
        # Header
        gr.Markdown("""
        # 🏦 TrustVoice Analytics
        ## Intelligent Complaint Analysis for Financial Services
        
        Ask questions about customer complaints across Credit Cards, Personal Loans, BNPL, Savings Accounts, and Money Transfers.
        
        **Example Questions:**
        - Why are people unhappy with Buy Now, Pay Later?
        - What are the main issues with Credit Cards?
        - What are the most common customer service issues?
        - What emerging trends should product managers be aware of?
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation History",
                    height=400,
                    show_label=True
                )
                
                # Input area
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Ask a question about customer complaints:",
                        placeholder="e.g., Why are people unhappy with BNPL?",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)
                
                # Clear button
                clear_btn = gr.Button("Clear Conversation", variant="secondary")
            
            with gr.Column(scale=1):
                # Answer display
                answer_display = gr.Markdown(
                    label="AI Answer",
                    value="",
                    elem_classes=["answer-box"]
                )
                
                # Sources display
                sources_display = gr.Markdown(
                    label="Sources Used",
                    value="",
                    elem_classes=["source-box"]
                )
        
        # Footer
        gr.Markdown("""
        ---
        **About:** This system uses Retrieval-Augmented Generation (RAG) to analyze CFPB complaint data and provide insights for CrediTrust Financial's internal teams.
        
        **Data Source:** Consumer Financial Protection Bureau (CFPB) complaint database
        **Model:** Mistral AI Magistral-Small-2506 via Hugging Face Inference API
        """)
        
        # Event handlers
        submit_btn.click(
            fn=query_rag,
            inputs=[question_input, chatbot],
            outputs=[answer_display, sources_display, chatbot],
            api_name="ask_question"
        )
        
        question_input.submit(
            fn=query_rag,
            inputs=[question_input, chatbot],
            outputs=[answer_display, sources_display, chatbot],
            api_name="ask_question_enter"
        )
        
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[answer_display, sources_display, chatbot],
            api_name="clear_chat"
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        title="TrustVoice Analytics"
    )
