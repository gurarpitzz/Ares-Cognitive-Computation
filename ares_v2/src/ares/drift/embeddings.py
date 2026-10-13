"""Sentence Embedding Extractor and Disk Cache for Drift and Mutation Analysis (M06)."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ares.common.paths import ARES_ROOT


class SentenceEmbedder:
    """Extracts raw and L2-normalized dense sentence embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def encode(
        self,
        texts: List[str],
        batch_size: int = 128,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """Encode a list of sentences into dense vectors."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)
        
    def encode_pairs(
        self,
        s1_list: List[str],
        s2_list: List[str],
        batch_size: int = 128,
        normalize_embeddings: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode paired sentences (s1, s2) and compute perturbation vector epsilon = s2 - s1."""
        emb1 = self.encode(s1_list, batch_size=batch_size, normalize_embeddings=normalize_embeddings)
        emb2 = self.encode(s2_list, batch_size=batch_size, normalize_embeddings=normalize_embeddings)
        diff = emb2 - emb1
        return emb1, emb2, diff
