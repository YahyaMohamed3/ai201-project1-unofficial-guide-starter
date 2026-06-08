from ingestion import load_documents

CHUNK_SIZE = 800
OVERLAP = 100

def chunk_text(text, source):
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) > 0:
            chunks.append({
                "chunk_id": f"{source}_chunk_{chunk_index}",
                "source": source,
                "chunk_index": chunk_index,
                "text": chunk
            })
            chunk_index += 1
        start += CHUNK_SIZE - OVERLAP
    return chunks

def chunk_documents(documents):
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_text(doc["text"], doc["source"])
        all_chunks.extend(doc_chunks)
        print(f"{doc['source']}: {len(doc_chunks)} chunks")
    return all_chunks

if __name__ == "__main__":
    docs = load_documents()
    print()
    chunks = chunk_documents(docs)
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"\n--- Sample chunk ---")
    print(f"ID: {chunks[5]['chunk_id']}")
    print(f"Source: {chunks[5]['source']}")
    print(f"Text:\n{chunks[5]['text']}")
