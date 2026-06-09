import gradio as gr
from groq import Groq
from retrieve import retrieve_hybrid
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "launchmap"
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)

SOURCES = [
    "All Sources",
    "aws_cloud_practitioner.txt",
    "aws_solutions_architect.txt",
    "behavioral_interview_guide.txt",
    "codepath_applied_ai.txt",
    "codepath_tip.txt",
    "coding_interview_university.txt",
    "interview_guide.txt",
    "resume_guide.txt",
    "tech_interview_cheatsheet.txt",
    "tech_interview_handbook.txt"
]

def handle_query(question, source_filter):
    if not question.strip():
        yield "Please enter a question.", ""
        return

    # Metadata filtering
    if source_filter and source_filter != "All Sources":
        q_vec = model.encode(question).tolist()
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=5,
            where={"source": source_filter},
            include=["documents", "metadatas", "distances"]
        )
        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i]
            })
    else:
        chunks = retrieve_hybrid(question)

    context = "\n\n".join([c["text"] for c in chunks])
    sources = "\n".join(f"• {s}" for s in list(set([c["source"] for c in chunks])))

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for CS students preparing "
                    "for internships and technical interviews. "
                    "Answer ONLY using the provided context. "
                    "If the context does not contain enough information, say: "
                    "'I don't have enough information on that in my documents.' "
                    "Do not use your own knowledge or make anything up. "
                    "Format your answer using markdown."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        stream=True
    )

    answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            answer += delta
            yield answer, sources

with gr.Blocks(title="LaunchMap") as demo:
    gr.Markdown("# LaunchMap")
    gr.Markdown("### AI-powered internship prep guide for CS students")

    with gr.Row():
        inp = gr.Textbox(
            label="Your Question",
            placeholder="e.g. What is the difference between AWS Cloud Practitioner and Solutions Architect?",
            lines=2
        )
        source_filter = gr.Dropdown(
            choices=SOURCES,
            value="All Sources",
            label="Filter by Source"
        )

    btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        answer = gr.Markdown(label="Answer")
        sources = gr.Textbox(label="Sources", lines=10)

    btn.click(handle_query, inputs=[inp, source_filter], outputs=[answer, sources])
    inp.submit(handle_query, inputs=[inp, source_filter], outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()