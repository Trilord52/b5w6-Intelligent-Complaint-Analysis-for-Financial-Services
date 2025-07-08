"""
chunk_embed_index.py
Script to chunk complaint narratives, generate embeddings, and index them in FAISS.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os
import pickle
import torch

from chunking import chunk_text
from embedding import Embedder
from vector_store import VectorStore

def main():
    """
    Memory-efficient, resumable pipeline for chunking, embedding, and indexing complaint narratives.
    Processes the cleaned CSV in batches, checkpointing after each batch.
    """
    # Parameters
    input_path = Path('data/complaints_processed.csv')
    index_path = Path('vector_store/faiss_index.idx')
    metadata_path = Path('vector_store/metadata.pkl')
    progress_path = Path('vector_store/progress.chkpt')
    chunk_size = 100  # words
    overlap = 20      # words
    embed_batch_size = 32  # Conservative for 8GB RAM/GTX 1050
    csv_batch_size = 1000  # Number of rows per CSV chunk
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    # Create output directory
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine where to resume from
    start_batch = 0
    if progress_path.exists():
        with open(progress_path, 'rb') as f:
            start_batch = pickle.load(f)
        print(f"Resuming from batch {start_batch}...")
    else:
        print("Starting from scratch...")

    # Prepare embedder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}, embedding batch size: {embed_batch_size}")
    embedder = Embedder(model_name=model_name, batch_size=embed_batch_size, device=device, torch_dtype='float16')

    # Prepare vector store (load if exists, else create new)
    if index_path.exists() and metadata_path.exists():
        store = VectorStore.load(str(index_path), str(metadata_path))
        print(f"Loaded existing index and metadata with {len(store.metadata)} entries.")
    else:
        # We'll determine dim after first batch
        store = None

    # Process CSV in batches
    reader = pd.read_csv(input_path, chunksize=csv_batch_size)
    for batch_idx, df in enumerate(reader):
        if batch_idx < start_batch:
            continue  # Skip already-processed batches
        print(f"Processing batch {batch_idx} (rows {batch_idx*csv_batch_size} to {(batch_idx+1)*csv_batch_size-1})...")
        all_chunks = []
        all_metadata = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = str(row['Consumer complaint narrative_cleaned'])
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    'complaint_id': row['Complaint ID'],
                    'product': row['Product'],
                    'chunk': chunk
                })
        if not all_chunks:
            print(f"No chunks in batch {batch_idx}, skipping.")
            with open(progress_path, 'wb') as f:
                pickle.dump(batch_idx + 1, f)
            continue
        # Embed
        print(f"Embedding {len(all_chunks)} chunks...")
        embeddings = embedder.embed_chunks(all_chunks)
        # Initialize vector store if first batch
        if store is None:
            store = VectorStore(dim=embeddings.shape[1])
        # Add to index
        store.add(embeddings, all_metadata)
        # Save index and metadata (checkpoint)
        store.save(str(index_path), str(metadata_path))
        with open(progress_path, 'wb') as f:
            pickle.dump(batch_idx + 1, f)
        print(f"Batch {batch_idx} processed and checkpointed. Total indexed: {len(store.metadata)}")
    # Cleanup progress file
    if progress_path.exists():
        os.remove(progress_path)
    print(f"All batches processed. Final index and metadata saved to {index_path} and {metadata_path}.")

if __name__ == "__main__":
    main() 