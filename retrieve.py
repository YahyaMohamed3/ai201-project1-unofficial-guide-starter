import chromadb
from sentence_transformers import SentenceTransformer

VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "launchmap"
TOP_K = 5



# Load model and collection once
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
collection = client.get_collection(name=COLLECTION_NAME)


def retrieve(query, top_k=TOP_K):
    # Step 1: Embed the query
    query_embedding = model.encode(query).tolist()

    # Step 2: search in ChromaDB for k closest chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Step 3: Format results
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i]
        })
    return chunks

if __name__ == "__main__":
    query = "What is the difference between AWS Cloud Practitioner and Solutions Architect?"
    print(f"Query: {query}\n")
    results = retrieve(query)
    for i, chunk in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Source: {chunk['source']}")
        print(f"Distance: {chunk['distance']:.4f}")
        print(f"Text: {chunk['text'][:300]}")
        print()