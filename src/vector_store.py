"""
vector_store.py
Module for storing and retrieving embeddings using FAISS.
"""
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, dim: int):
        """
        Initializes a FAISS index for storing embeddings.
        Args:
            dim (int): Dimension of the embeddings.
        """
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []  # List of dicts, one per embedding

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """
        Adds embeddings and their metadata to the index.
        Args:
            embeddings (np.ndarray): Embedding vectors.
            metadatas (List[Dict]): Metadata for each embedding.
        """
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadatas)

    def save(self, index_path: str, metadata_path: str):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    @staticmethod
    def load(index_path: str, metadata_path: str):
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        store = VectorStore(index.d)
        store.index = index
        store.metadata = metadata
        return store 