import os
import chromadb
from sentence_transformers import SentenceTransformer
from chunking import chunk_documents
from ingestion import load_documents

VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "launchmap"

def build_vectorstore():
    # Step 1: Load and chunk documents
    docs = load_documents()
    print()
    chunks = chunk_documents(docs)
    print(f"\nTotal chunks to embed: {len(chunks)}")

    # Step 2: Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 3: Set up ChromaDB on disk
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)

    # Step 4: Delete old collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection: {COLLECTION_NAME}")
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    # Step 5: Embed all chunks
    print("\nEmbedding chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Step 6: Store in ChromaDB with metadata
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{
            "source": c["source"],
            "chunk_index": c["chunk_index"]
        } for c in chunks]
    )

    print(f"\nStored {collection.count()} chunks in ChromaDB")
    return collection

if __name__ == "__main__":
    collection = build_vectorstore()
    print("\nVectorstore built successfully.")
    # Show one sample from the store
    sample = collection.get(ids=["aws_cloud_practitioner.txt_chunk_0"])
    print(f"\n--- Sample stored chunk ---")
    print(f"ID: {sample['ids'][0]}")
    print(f"Source: {sample['metadatas'][0]['source']}")
    print(f"Text:\n{sample['documents'][0][:300]}")
