"""
embedding.py
Module for generating embeddings from text chunks using sentence-transformers.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', batch_size: int = 64, device: str = None, torch_dtype: str = None):
        """
        Initializes the embedding model on the specified device, with optional mixed precision.
        Args:
            model_name (str): Name of the sentence transformer model.
            batch_size (int): Batch size for embedding.
            device (str): 'cuda', 'cpu', or None to auto-detect.
            torch_dtype (str): 'float16' for mixed precision, or None for default.
        """
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing embedder on {device}")
        kwargs = {}
        if torch_dtype:
            kwargs['torch_dtype'] = torch.float16 if torch_dtype == 'float16' else torch.float32
        self.model = SentenceTransformer(model_name, device=device, model_kwargs=kwargs)
        if torch_dtype == 'float16':
            self.model.half()
        self.batch_size = batch_size
        self.device = device

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of text chunks using the specified device and batch size.
        Args:
            chunks (List[str]): List of text chunks.
        Returns:
            np.ndarray: Array of embeddings.
        """
        return np.array(self.model.encode(
            chunks,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )) 