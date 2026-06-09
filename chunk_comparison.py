import chromadb
import os
from sentence_transformers import SentenceTransformer
from ingestion import load_documents

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, source, chunk_size, overlap):
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 0:
            chunks.append({
                "chunk_id": f"{source}_c{chunk_size}_chunk_{chunk_index}",
                "source": source,
                "chunk_index": chunk_index,
                "text": chunk
            })
            chunk_index += 1
        start += chunk_size - overlap
    return chunks

def build_temp_collection(chunk_size, overlap, client):
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc["text"], doc["source"], chunk_size, overlap))

    name = f"launchmap_c{chunk_size}"
    try:
        client.delete_collection(name)
    except:
        pass

    collection = client.create_collection(name)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts)
    collection.add(
        ids=[c["chunk_id"] for c in all_chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"source": c["source"]} for c in all_chunks]
    )
    print(f"Built collection with chunk_size={chunk_size}: {len(all_chunks)} chunks")
    return collection

def test_query(collection, query, chunk_size):
    q_vec = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    print(f"\n  chunk_size={chunk_size} | Query: {query[:60]}")
    for i in range(len(results["ids"][0])):
        src = results["metadatas"][0][i]["source"]
        dist = results["distances"][0][i]
        text = results["documents"][0][i][:150]
        print(f"  [{dist:.4f}] {src}: {text}")

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="vectorstore")

    print("Building collections...")
    col_400 = build_temp_collection(400, 50, client)
    col_800 = build_temp_collection(800, 100, client)
    col_1200 = build_temp_collection(1200, 150, client)

    queries = [
        "What should I do after I finish coding in an interview?",
        "What is the difference between AWS Cloud Practitioner and Solutions Architect?",
        "What does CodePath TIP teach?"
    ]

    print("\n" + "="*60)
    print("CHUNKING STRATEGY COMPARISON")
    print("="*60)

    for query in queries:
        print(f"\nQUERY: {query}")
        print("-"*60)
        test_query(col_400, query, 400)
        test_query(col_800, query, 800)
        test_query(col_1200, query, 1200)