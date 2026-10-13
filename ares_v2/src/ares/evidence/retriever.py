"""Retrieval operators for candidate evidence extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

@dataclass
class EvidenceRecord:
    doc_id: str
    text: str
    source: str
    score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None

class BaseRetriever(ABC):
    """Abstract base class for evidence retrieval."""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Index documents (each dict must contain 'id' and 'text')."""
        pass
        
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[EvidenceRecord]:
        """Retrieve top_k most relevant evidence passages for the given query."""
        pass

class BM25Retriever(BaseRetriever):
    """Lexical BM25 retriever based on Okapi BM25."""
    
    def __init__(self):
        self.documents = []
        self.doc_ids = []
        self.sources = []
        self.bm25: Optional[BM25Okapi] = None
        
    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        self.documents = [doc["text"] for doc in documents]
        self.doc_ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]
        self.sources = [doc.get("source", "corpus") for doc in documents]
        corpus_tokens = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(corpus_tokens)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[EvidenceRecord]:
        if not self.bm25 or len(self.documents) == 0:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        records = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] <= 0.0 and rank > 1:
                # Discard zero-overlap scores beyond first
                continue
            records.append(
                EvidenceRecord(
                    doc_id=self.doc_ids[idx],
                    text=self.documents[idx],
                    source=self.sources[idx],
                    score=float(scores[idx]),
                    rank=rank
                )
            )
        return records

class DenseRetriever(BaseRetriever):
    """Dense vector retriever using SentenceTransformers and FAISS."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        import faiss
        
        self.model = SentenceTransformer(model_name, device=device)
        self.documents = []
        self.doc_ids = []
        self.sources = []
        self.index = None
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        import faiss
        self.documents = [doc["text"] for doc in documents]
        self.doc_ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]
        self.sources = [doc.get("source", "corpus") for doc in documents]
        
        if len(self.documents) == 0:
            return
            
        embeddings = self.model.encode(
            self.documents,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True
        ).astype(np.float32)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[EvidenceRecord]:
        if self.index is None or len(self.documents) == 0:
            return []
            
        query_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(query_emb, top_k)
        
        records = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(self.documents):
                continue
            records.append(
                EvidenceRecord(
                    doc_id=self.doc_ids[idx],
                    text=self.documents[idx],
                    source=self.sources[idx],
                    score=float(score),
                    rank=rank
                )
            )
        return records

class HybridRetriever(BaseRetriever):
    """Hybrid BM25 + Dense retriever using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, bm25_retriever: BM25Retriever, dense_retriever: DenseRetriever, rrf_k: int = 60):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.doc_map: Dict[str, Dict[str, Any]] = {}
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            self.doc_map[doc.get("id", str(len(self.doc_map)))] = doc
        self.bm25.add_documents(documents)
        self.dense.add_documents(documents)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[EvidenceRecord]:
        bm25_results = self.bm25.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)
        
        rrf_scores: Dict[str, float] = {}
        text_map: Dict[str, str] = {}
        source_map: Dict[str, str] = {}
        
        for rec in bm25_results:
            rrf_scores[rec.doc_id] = rrf_scores.get(rec.doc_id, 0.0) + (1.0 / (self.rrf_k + rec.rank))
            text_map[rec.doc_id] = rec.text
            source_map[rec.doc_id] = rec.source
            
        for rec in dense_results:
            rrf_scores[rec.doc_id] = rrf_scores.get(rec.doc_id, 0.0) + (1.0 / (self.rrf_k + rec.rank))
            text_map[rec.doc_id] = rec.text
            source_map[rec.doc_id] = rec.source
            
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        records = []
        for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
            records.append(
                EvidenceRecord(
                    doc_id=doc_id,
                    text=text_map.get(doc_id, ""),
                    source=source_map.get(doc_id, "corpus"),
                    score=float(score),
                    rank=rank
                )
            )
        return records
