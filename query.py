import os 
from dotenv import load_dotenv
from groq import Groq
from retrieve import retrieve

load_dotenv()


client = Groq(api_key = os.getenv("GROQ_API_KEY"))


def ask(question):
    # Step 1: Retrieve relavnt chunks
    chunks = retrieve(question)

    # Step 2: combine chunk texts into one context string
    context = "\n\n".join([c["text"] for c in chunks])

    # Step 3: collect source filenames, remove duplicates
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for CS students preparing "
                    "for internships and technical interviews. "
                    "Answer ONLY using the provided context. "
                    "If the context does not contain enough information "
                    "to answer the question, say: "
                    "'I don't have enough information on that in my documents.' "
                    "Do not use your own knowledge or make anything up."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    # Step 5: extract answer text
    answer = response.choices[0].message.content

    # Step 6: return answer and resources
    return {
        "answer": answer,
        "sources": list(set([c["source"] for c in chunks])),
        "chunks" : chunks
    }

if __name__ == "__main__":
    question = "What is the difference between AWS Cloud Practitioner and Solutions Architect?"
    result = ask(question)
    print(f"Question: {question}\n")
    print(f"Answer: {result['answer']}\n")
    print(f"Sources: {', '.join(result['sources'])}")