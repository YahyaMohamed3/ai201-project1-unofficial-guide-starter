import gradio as gr
from groq import Groq
from retrieve import retrieve_hybrid
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def handle_query(question):
    if not question.strip():
        yield "Please enter a question.", ""
        return

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
                    "Format your answer using markdown: use bold, bullet points, and headers where helpful."
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
    gr.Markdown("Ask anything about technical interviews, AWS certifications, CodePath, or resume advice.")

    inp = gr.Textbox(
        label="Your Question",
        placeholder="e.g. How do I write my resume projects section?",
        lines=2
    )

    btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        answer = gr.Markdown(label="Answer")
        sources = gr.Textbox(label="Sources", lines=10)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()