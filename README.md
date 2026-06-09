**How source attribution is surfaced in the response:** Source filenames are
collected programmatically from the metadata of every retrieved chunk, deduplicated,
and displayed in a separate Sources panel in the Gradio UI. This is done in code
rather than relying on the LLM to cite sources, which guarantees attribution
appears on every response regardless of what the model outputs.

---

## Retrieval Test Results

**Query 1:** "What is the difference between AWS Cloud Practitioner and Solutions Architect?"

Top returned chunks:
- aws_solutions_architect.txt (distance: 0.6907) — overview of Solutions Architect cert
- aws_cloud_practitioner.txt (distance: 0.7909) — next steps after Cloud Practitioner
- aws_solutions_architect.txt (distance: 0.8719) — next certifications after Solutions Architect
- aws_cloud_practitioner.txt (distance: 0.9250) — Cloud Practitioner overview
- aws_solutions_architect.txt (distance: 1.0233) — prep steps for Solutions Architect

Why these chunks are relevant: Both AWS certification files were retrieved because
the query directly asks about these two certifications. The chunks contain exam
details, target audiences, and cost differences that directly answer the comparison
question. The semantic similarity correctly identified both files as relevant despite
the query not containing exact phrases from the documents.

**Query 2:** "What does CodePath TIP teach?"

Top returned chunks:
- codepath_tip.txt (distance: 0.6444) — TIP course overview and curriculum
- codepath_applied_ai.txt (distance: 0.8467) — Applied AI pathway overview

Why these chunks are relevant: The top result from codepath_tip.txt directly
contains the course curriculum including UMPIRE, algorithms, and data structures.
The second result from codepath_applied_ai.txt is partially relevant as it covers
a related CodePath program.

**Query 3:** "What should I do after I finish coding in an interview?"

Top returned chunks:
- coding_interview_university.txt (distance: 0.7936) — common mistakes to avoid
- tech_interview_cheatsheet.txt (distance: 0.8058) — general interview behavior
- tech_interview_cheatsheet.txt (distance: 0.8170) — before the interview steps

Why retrieval partially failed: The relevant content (Step 5 - Review and Test,
Step 6 - End of Interview) exists in tech_interview_cheatsheet.txt but was split
across a chunk boundary. Neither half chunk contained enough semantic signal to
rank highly. The phrasing "after you finish coding" does not closely match the
document language "do not announce you are done."

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What is the difference between AWS Cloud Practitioner and Solutions Architect? | Cloud Practitioner is foundational $100 90min no experience required. Solutions Architect is associate level $150 130min recommends 1 year experience. | Correctly described both certs with focus, target audience, experience requirements, and cost differences. | Relevant | Accurate |
| 2 | What does CodePath TIP teach and how do I apply? | Free 10-week course covering UMPIRE, algorithms, data structures, Big O. Apply via questionnaire and HackerRank. | Correctly listed all curriculum topics, three levels, eligibility, and two-step application process. | Relevant | Accurate |
| 3 | What should I do after I finish coding in an interview? | Do not announce done. Scan for bugs, brainstorm edge cases, step through code, reiterate complexity. | Returned "I don't have enough information on that in my documents." | Off-target | Inaccurate |
| 4 | What should I put in the projects section of my resume? | At least 2 projects linked to GitHub, specific contributions and technologies used. | Correctly stated at least 2 projects, link to GitHub, specific contributions, technologies used, with example. | Relevant | Accurate |
| 5 | What programming language should I use for coding interviews? | Use the language you know best. Python, Java, C++, JavaScript most common. Python recommended. | Correctly recommended Python, Java, C++, JavaScript with advice to use familiar language. | Relevant | Accurate |

---

## Failure Case Analysis

**Question that failed:** "What should I do after I finish coding in an interview?"

**What the system returned:** "I don't have enough information on that in my
documents." — despite the answer existing in tech_interview_cheatsheet.txt.

**Root cause (tied to a specific pipeline stage):** The failure occurred at the
chunking and retrieval stages together. The answer is in Steps 5 and 6 of the
cheatsheet (Review and Test, End of Interview) but these steps were split across
a chunk boundary at 800 characters. Each half-chunk lacked enough standalone
semantic signal to rank highly. Additionally, the query uses the phrase "after
I finish coding" while the document uses "do not announce you are done" — the
semantic distance between these phrasings was large enough (distance 0.80+) that
the relevant chunk was not returned in the top 5. The hybrid BM25 search also
failed because the keywords "finish coding" do not appear verbatim in the chunk.

**What you would change to fix it:** Increase chunk size to 1200 characters —
the chunking comparison experiment showed that 1200-char chunks achieved a
distance of 0.6985 for this query versus 0.7936 for 800-char chunks, keeping
more of the step-by-step content together. Alternatively, paragraph-based
chunking would keep all 6 interview steps in fewer, more coherent chunks.

---

## Hybrid Search Comparison

The system implements hybrid search combining semantic vector similarity
(all-MiniLM-L6-v2 via ChromaDB) with BM25 keyword search (rank-bm25).
Scores are combined with equal weight: 0.5 * semantic_score + 0.5 * bm25_score.
Semantic scores are converted from distances using 1/(1+distance).

Results across 3 queries:

**Query: "What should I do after I finish coding in an interview?"**
- Semantic only: best distance 0.7936 (coding_interview_university.txt) — off-target
- Hybrid: best score 0.7448 (coding_interview_university.txt) — still off-target
- Neither method retrieved the correct chunk due to chunk boundary split

**Query: "What is the difference between AWS Cloud Practitioner and Solutions Architect?"**
- Semantic only: best distance 0.6907 (aws_solutions_architect.txt) — relevant
- Hybrid: best score 0.6618 (tech_interview_cheatsheet.txt) — less relevant
- Semantic performed better for this query

**Query: "What does CodePath TIP teach?"**
- Semantic only: best distance 0.6444 (codepath_tip.txt) — relevant
- Hybrid: best score 0.6444 (codepath_tip.txt) — same top result
- Both methods performed equally

**Conclusion:** Semantic search outperformed hybrid search on 2 of 3 queries.
BM25 keyword matching helped surface exact terminology in some cases but also
introduced noise by boosting chunks with incidental keyword matches. The 800-char
chunk size was the more significant factor in retrieval quality than the search
method.

---

## Chunking Strategy Comparison

Three chunk sizes were tested on the same 3 queries:

| Query | 400-char best distance | 800-char best distance | 1200-char best distance | Winner |
|-------|----------------------|----------------------|------------------------|--------|
| After finish coding | 0.7886 | 0.7936 | 0.6985 | 1200 |
| AWS cert difference | 0.5917 | 0.6907 | 0.7784 | 400 |
| CodePath TIP | 0.6444 | 0.9929 | 1.0633 | 400 |

400-char chunks won on 2 of 3 queries by producing tighter, more focused
embeddings that matched specific topics precisely. 1200-char chunks won on
the "after coding" query because the relevant content was spread across multiple
steps that got split at 800 characters — larger chunks kept more steps together.

Overall 400-char chunking performed best for this corpus of structured guides,
suggesting that the documents contain enough dense, topic-specific content per
paragraph that smaller chunks retrieve more precisely. The one exception is
step-by-step content where larger chunks prevent boundary splits.

---

## Query Interface

The interface is built with Gradio and runs at http://localhost:7860.

**Input fields:**
- Your Question: a text box where the user types any natural language question
- Filter by Source: a dropdown to restrict search to a specific document or search all sources
- Ask button: submits the question

**Output fields:**
- Chat window: displays the full conversation history with streaming responses in markdown
- Sources panel: shows which document files the answer was drawn from

**Sample interaction transcript:**

User: What is the difference between AWS Cloud Practitioner and Solutions Architect?

System: The main difference between the two certifications is their focus and
target audience.

**AWS Cloud Practitioner:**
- Validates foundational, high-level understanding of AWS Cloud
- Ideal for individuals with no prior IT or cloud experience
- Cost: $100 USD, 90 minutes, Foundational category

**AWS Certified Solutions Architect - Associate:**
- Focuses on design of cost and performance optimized solutions
- Requires at least 1 year of hands-on experience
- Cost: $150 USD, 130 minutes, Associate category

Sources: aws_solutions_architect.txt, aws_cloud_practitioner.txt

---

## Spec Reflection

**One way the spec helped during implementation:** The chunking strategy section
of planning.md forced me to think about chunk size before writing any code. When
I chose 800 characters I had to justify why that fit the document structure —
which made me notice that my documents were structured guides, not short reviews,
and needed larger chunks than review-style text. This decision directly shaped
the ingestion pipeline and made the retrieval results more coherent than if I
had used a default size.

**One way implementation diverged from the spec:** The planning.md specified that
hybrid search would improve retrieval quality over semantic search alone. In
practice, the chunking comparison experiment showed that chunk size was a more
significant factor than search method — 400-char chunks outperformed 800-char
chunks on 2 of 3 queries regardless of whether hybrid or semantic search was
used. The spec assumed hybrid search would be the main performance lever, but
the actual bottleneck was chunk boundary splits.

---

## AI Usage

**Instance 1**

- *What I gave the AI:* My planning.md chunking strategy section specifying
  800-character chunks with 100-character overlap, and the requirement to load
  .txt files from a documents/ folder and strip SOURCE_TITLE/SOURCE_URL headers.
- *What it produced:* A complete ingestion.py with regex cleaning and a
  chunking.py with a character-based sliding window function.
- *What I changed or overrode:* The initial ingestion.py used "filename" as
  the dict key instead of "source", which caused a KeyError in chunking.py.
  I identified the bug by running python -c to inspect the dict keys, then
  directed Claude to fix the key name consistently across both files. I also
  added the print statement format myself to match the output style I wanted.

**Instance 2**

- *What I gave the AI:* The retrieval approach section from planning.md
  specifying all-MiniLM-L6-v2, ChromaDB, top-k=5, and the requirement to
  store source filename as metadata for attribution.
- *What it produced:* embed.py and retrieve.py with ChromaDB PersistentClient,
  collection.add() with embeddings and metadatas, and a retrieve() function.
- *What I changed or overrode:* The generated embed.py passed embeddings as
  a numpy array directly to ChromaDB which caused a type error. I identified
  this by reading the error message and directed Claude to add .tolist() to
  convert the numpy array. I also added the sample chunk verification at the
  bottom of embed.py myself to confirm the store was working correctly.