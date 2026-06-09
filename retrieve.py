import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "launchmap"
TOP_K = 5

# Load model and collection once
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

# Load all chunks once for BM25 index
all_data = collection.get(include=["documents", "metadatas"])
all_texts = all_data["documents"]
all_metadatas = all_data["metadatas"]
all_ids = all_data["ids"]

# Build BM25 index from tokenized chunks
tokenized = [text.lower().split() for text in all_texts]
bm25 = BM25Okapi(tokenized)


def retrieve(query, top_k=TOP_K):
    # Step 1: Embed the query
    query_embedding = model.encode(query).tolist()

    # Step 2: Search ChromaDB for k closest chunks
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


def retrieve_hybrid(query, top_k=TOP_K):
    # Step 1: Get semantic scores for all chunks
    query_embedding = model.encode(query).tolist()
    sem_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=len(all_texts),
        include=["documents", "metadatas", "distances"]
    )

    # Step 2: Convert distances to scores (lower distance = higher score)
    sem_scores = {}
    for i, doc_id in enumerate(sem_results["ids"][0]):
        sem_scores[doc_id] = 1 / (1 + sem_results["distances"][0][i])

    # Step 3: Get BM25 keyword scores and normalize to 0-1
    tokenized_query = query.lower().split()
    bm25_raw = bm25.get_scores(tokenized_query)
    max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_scores = {all_ids[i]: bm25_raw[i] / max_bm25 for i in range(len(all_ids))}

    # Step 4: Combine semantic and BM25 scores with equal weight
    combined = {}
    for doc_id in all_ids:
        sem = sem_scores.get(doc_id, 0)
        bm25_s = bm25_scores.get(doc_id, 0)
        combined[doc_id] = 0.5 * sem + 0.5 * bm25_s

    # Step 5: Sort by combined score and take top k
    top_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]

    # Step 6: Fetch full chunk data and format results
    fetched = collection.get(ids=top_ids, include=["documents", "metadatas"])
    id_to_data = {
        fetched["ids"][i]: {
            "text": fetched["documents"][i],
            "source": fetched["metadatas"][i]["source"]
        }
        for i in range(len(fetched["ids"]))
    }

    chunks = []
    for doc_id in top_ids:
        chunks.append({
            "text": id_to_data[doc_id]["text"],
            "source": id_to_data[doc_id]["source"],
            "score": combined[doc_id]
        })
    return chunks


if __name__ == "__main__":
    query = "What should I do after I finish coding in an interview?"
    print(f"Query: {query}\n")

    print("SEMANTIC ONLY")
    results = retrieve(query)
    for i, chunk in enumerate(results):
        print(f"Result {i+1}")
        print(f"Source: {chunk['source']}")
        print(f"Distance: {chunk['distance']:.4f}")
        print(f"Text: {chunk['text'][:300]}")
        print()

    print("HYBRID (Semantic + BM25)")
    results = retrieve_hybrid(query)
    for i, chunk in enumerate(results):
        print(f"Result {i+1}")
        print(f"Source: {chunk['source']}")
        print(f"Score: {chunk['score']:.4f}")
        print(f"Text: {chunk['text'][:300]}")
        print()