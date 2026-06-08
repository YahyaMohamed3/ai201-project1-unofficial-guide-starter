import os
import re

DOCUMENTS_DIR = "documents"

def clean_text(text):
    text = re.sub(r'SOURCE_TITLE:.*\n', '', text)
    text = re.sub(r'SOURCE_URL:.*\n', '', text)
    text = re.sub(r'SOURCE_TYPE:.*\n', '', text)
    text = re.sub(r'CONTENT:\s*\n', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def load_documents():
    documents = []
    for filename in sorted(os.listdir(DOCUMENTS_DIR)):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCUMENTS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()
            cleaned = clean_text(raw_text)
            if cleaned:
                documents.append({
                    "source": filename,
                    "text": cleaned
                })
                print(f"Loaded: {filename} ({len(cleaned)} chars)")
    return documents

if __name__ == "__main__":
    docs = load_documents()
    print(f"\nTotal documents loaded: {len(docs)}")
    print(f"\n--- Sample ({docs[0]['source']}) ---")
    print(docs[0]['text'][:500])
