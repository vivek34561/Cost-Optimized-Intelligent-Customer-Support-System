"""
Build FAISS RAG Index
======================

Command-line script to build FAISS index from customer support dataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import argparse
from src.faiss_index_builder import FAISSIndexBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index for RAG chatbot"
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents to index (useful for testing)'
    )
    
    args = parser.parse_args()
    
    # Build index
    builder = FAISSIndexBuilder()
    builder.build(limit=args.limit)
    
    print("\n✅ Done! You can now:")
    print("  1. Test retrieval: python src/retriever.py")
    print("  2. Test chatbot: python src/main.py")
    print("  3. Interactive mode: python src/main.py interactive")


if __name__ == "__main__":
    main()
