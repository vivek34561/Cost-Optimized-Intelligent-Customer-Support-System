"""
FAISS Index Builder
====================

Builds and populates FAISS vector index from the customer support dataset.
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

from src.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    PROJECT_ROOT
)


class FAISSIndexBuilder:
    """Builds FAISS index from customer support dataset"""
    
    def __init__(self):
        """Initialize FAISS and HuggingFace model"""
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.dimension = EMBEDDING_DIMENSION
        self.index = None
        self.metadata = []
    
    def create_index(self):
        """
        Create FAISS index (IndexFlatIP for cosine similarity via normalization)
        """
        print(f"Creating FAISS index with dimension {self.dimension}...")
        
        # Use IndexFlatIP (Inner Product) with normalized vectors for cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        
        print(f"  ✓ FAISS index created successfully")
        return self.index
    
    def load_dataset(self, limit=None):
        """
        Load customer support dataset from HuggingFace hub
        
        Args:
            limit: Optional limit on number of rows to load
            
        Returns:
            DataFrame with dataset
        """
        from huggingface_hub import hf_hub_download
        
        print(f"Loading dataset from HuggingFace hub...")
        
        # Download from HuggingFace
        try:
            csv_path = hf_hub_download(
                repo_id="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                filename="Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
                repo_type="dataset"
            )
            print(f"  ✓ Downloaded to {csv_path}")
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download dataset from HuggingFace: {e}\n"
                f"Please check your internet connection or download manually from:\n"
                f"https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"
            )
        
        df = pd.read_csv(csv_path)
        
        if limit:
            df = df.head(limit)
            print(f"  ✓ Loaded {len(df)} rows (limited)")
        else:
            print(f"  ✓ Loaded {len(df)} rows")
        
        return df
    
    def prepare_documents(self, df):
        """
        Prepare documents for indexing
        
        Args:
            df: DataFrame with instruction and response columns
            
        Returns:
            List of document dicts
        """
        print(f"\nPreparing {len(df)} documents...")
        
        documents = []
        
        for idx, row in df.iterrows():
            # Create document text (instruction + response)
            doc_text = f"Question: {row['instruction']}\nAnswer: {row['response']}"
            
            # Metadata
            metadata = {
                'instruction': row['instruction'],
                'response': row['response'],
                'intent': row['intent'],
                'category': row['category'],
                'text': doc_text
            }
            
            # Add optional fields
            for col in ['tags', 'response_type']:
                if col in row and pd.notna(row[col]):
                    metadata[col] = row[col]
            
            documents.append({
                'id': f"doc_{idx}",
                'text': doc_text,
                'metadata': metadata
            })
        
        print(f"  ✓ Prepared {len(documents)} documents")
        return documents
    
    def create_embeddings(self, documents, batch_size=32):
        """
        Create embeddings using HuggingFace sentence-transformers
        
        Args:
            documents: List of document dicts
            batch_size: Batch size for encoding
            
        Returns:
            Documents with embeddings added
        """
        print(f"\nCreating embeddings for {len(documents)} documents...")
        print(f"  Model: {EMBEDDING_MODEL}")
        
        texts = [doc['text'] for doc in documents]
        
        # Encode all texts with progress bar
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add embeddings to documents
        for i, embedding in enumerate(embeddings):
            documents[i]['embedding'] = embedding.tolist()
        
        print(f"  ✓ Created {len(documents)} embeddings")
        return documents
    
    def index_documents(self, documents):
        """
        Index documents into FAISS
        
        Args:
            documents: List of dicts with 'id', 'embedding', 'metadata'
        """
        if self.index is None:
            self.create_index()
        
        print(f"\nIndexing {len(documents)} documents into FAISS...")
        
        # Extract embeddings and metadata
        embeddings = np.array([doc['embedding'] for doc in documents], dtype='float32')
        
        # Normalize vectors for cosine similarity with IndexFlatIP
        faiss.normalize_L2(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)
        
        # Store metadata separately (FAISS doesn't store metadata internally)
        self.metadata = [
            {
                'id': doc['id'],
                'metadata': doc['metadata']
            }
            for doc in documents
        ]
        
        print(f"\n✓ Indexing complete!")
        print(f"  Total vectors: {self.index.ntotal}")
        
        return self.index
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        print(f"\nSaving FAISS index...")
        
        # Create data directory if needed
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        print(f"  ✓ Index saved to {FAISS_INDEX_PATH}")
        
        # Save metadata
        with open(FAISS_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Metadata saved to {FAISS_METADATA_PATH}")
    
    def build(self, limit=None):
        """
        Complete pipeline: load data -> create embeddings -> index -> save
        
        Args:
            limit: Optional limit on number of documents to process
        """
        print("="*80)
        print("BUILDING FAISS INDEX")
        print("="*80 + "\n")
        
        # Step 1: Create index
        self.create_index()
        
        # Step 2: Load dataset
        df = self.load_dataset(limit=limit)
        
        # Step 3: Prepare documents
        documents = self.prepare_documents(df)
        
        # Step 4: Create embeddings
        documents = self.create_embeddings(documents)
        
        # Step 5: Index documents
        self.index_documents(documents)
        
        # Step 6: Save to disk
        self.save_index()
        
        print("\n" + "="*80)
        print("✅ FAISS INDEX BUILD COMPLETE")
        print("="*80)


if __name__ == "__main__":
    # Test with small sample
    builder = FAISSIndexBuilder()
    builder.build(limit=10)  # Test with 10 documents
