# embedding_vector.py
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class EmbeddingVector:
    def __init__(self, model_name='intfloat/multilingual-e5-large-instruct', device=None): 
        if device is None:
            device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
        self.embedder = SentenceTransformer(model_name, device=device)

    def build_embeddings(self, text_chunks, cache_path='embeddings.npy', batch_size=32):
        """
        Build embeddings for a list of text chunks using caching.
        """
        if os.path.exists(cache_path):
            print("Loading cached embeddings from", cache_path)
            embeddings = np.load(cache_path)
        else:
            print("Computing embeddings for text chunks...")
            embeddings = self.embedder.encode(text_chunks, show_progress_bar=True, batch_size=batch_size)
            np.save(cache_path, embeddings)
        return embeddings

    def build_faiss_index(self, embeddings, n_neighbors=32, ef_construction=200, ef_search=50):
        """
        Build and return a FAISS HNSW index from the provided embeddings.
        """
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(embedding_dim, n_neighbors)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(np.array(embeddings).astype("float32"))
        print("HNSW FAISS index built with", index.ntotal, "vectors.")
        return index
