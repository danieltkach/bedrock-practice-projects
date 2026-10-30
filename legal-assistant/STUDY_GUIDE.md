# Study Guide: how the Legal Assistant works

Goal: after reading this you can explain every part of the project, why it was built that way, what went
wrong, and what you would improve. Read in order; each section builds on the previous one.

---

## 1. The problem and the idea (60-second version)

**Problem.** A language model can't read 250 legal PDFs at once. Pasting them all into a prompt would
exceed the model's input limit, cost a fortune per question, and give worse answers (long prompts dilute attention).

**Idea: RAG (Retrieval-Augmented Generation).** Don't give the model everything. Give it only the few passages that
look relevant to this question:

1. **Retrieve**: find the most relevant passages from the documents.
2. **Augment**: put those passages in the prompt.
3. **Generate**: ask the model to answer using only those passages, and to cite them.

Analogy: an open-book exam. Retrieval is finding the right pages in the book; generation is writing the answer
from those pages.

---

## 2. Glossary (the terms you'll be asked about)

| Term | Plain meaning | In this project |
|---|---|---|
| **Token** | A piece of a word; models read and bill in tokens | `usage.input_tokens` / `output_tokens` in responses |
| **Chunk** | A small piece of a document, sized to search and fit in a prompt | 1,200 characters of one page |
| **Overlap** | Chunks share their edges so a sentence cut at a boundary survives whole in the neighbor | 200 characters |
| **Embedding** | A list of numbers representing the *meaning* of a text. Similar meanings give similar lists | 512 numbers per chunk from Titan Text Embeddings v2 |
| **Vector** | That list of numbers, treated as a point/arrow in space | one row of `indexes/<name>.npy` |
| **Cosine similarity** | How closely two vectors point the same way (1 = same direction). Our measure of "closest" | `vectors @ query_vector` |
| **Normalized** | Vectors rescaled to length 1, so a dot product equals cosine similarity | `normalize: True` in the embed call |
| **Index** | A searchable collection: the vectors plus the text and source of each chunk | `opinions`, `transcripts` |
| **Top-k** | Keep only the k best matches | k = 5 |
| **Grounding** | Forcing answers to come from supplied sources | the system prompt |
| **Citation** | A pointer from a claim to its source | `[2]` -> file name + page |
| **Hallucination** | The model stating something unsupported | e.g. the "per curiam" slip (section 8) |
| **Per curiam** | Latin "by the court": an opinion issued by the whole court, with no named author. (The assistant wrongly called an opinion this; Justice Barrett had signed it) | example of a hallucination |
| **Map-reduce summary** | Summarize pieces (map), then summarize the summaries (reduce) | `/summary` |
| **Converse API** | Bedrock's uniform chat interface across models | `runtime.converse(...)` |

---

## 3. The two phases

### Phase A: Ingest (run once per document set), `ingest.py` + `rag.py`

1. **Read** each PDF page by page (`read_pdf_pages`), keeping the page number.
2. **Chunk** each page into ~1,200-character pieces with 200 characters of overlap (`chunk_text`).
3. **Embed** each chunk by calling Amazon Bedrock's Titan model (`embed`). This happens **on AWS, through the
   Bedrock API**, once per chunk, in parallel with 8 threads. Why embed? Because computers can't compare
   *meanings* of sentences directly; they can compare numbers. Similar meanings land near each other.
4. **Save** two files per index: `indexes/<name>.npy` (the numbers) and `indexes/<name>.json` (the text + file + page).
   Row *i* of the vectors matches entry *i* of the JSON.

Why a page never gets mixed with another: so each citation points at a real, single page.

### Phase B: Ask (every question), `assistant.py`

1. **Rewrite** the question if there is history (`standalone_question`). "Who wrote it?" alone is meaningless for
   search; Haiku turns it into "Who wrote the opinion in Acheson Hotels v. Laufer?".
2. **Embed** the question with the same Titan model.
3. **Search** (`Index.search`): one matrix multiplication scores every chunk; take the best 5 across the active
   indexes (`retrieve`).
4. **Build the prompt**: numbered sources `[1]..[5]` with file and page, then the question.
5. **Generate** with Claude Sonnet 4.5 under a strict system prompt: answer only from the sources, cite `[n]`,
   say so when the sources don't contain the answer.
6. **Print** the answer and the source list; **append** to history for the next turn.

---

## 4. Code walkthrough

**`rag.py`**
- `read_pdf_pages(path)`: opens a PDF, returns `[(page_number, text)]`; collapses whitespace.
- `chunk_text(text, size, overlap)`: slides a window of `size` characters; moves start back by `overlap`; prefers
  cutting at a space so words aren't split.
- `embed(text)`: `invoke_model` to Titan with `{inputText, dimensions: 512, normalize: true}` and returns the list of
  512 floats.
- `embed_many(texts)`: same, in parallel with a thread pool (the calls wait on the network, so threads help).
- `Index`: loads the vectors and metadata; `search(vector, k)` does `vectors @ query`, sorts, returns the top k
  with score, source, page, text.

**`ingest.py`**: glue: folder of PDFs -> chunks -> vectors -> two files.

**`assistant.py`**
- `ask(model, system, messages)`: one call to the Converse API; returns only the text blocks (see lesson 2 below).
- `standalone_question`, `retrieve`, `answer`: steps 1 to 6 above.
- `summarize(part)`: finds a PDF by name part, splits its text into ~20k-character sections, summarizes each with
  Haiku (150 words), then asks Sonnet to merge the notes into an executive summary.
- `main()`: the chat loop and the commands `/indexes`, `/use`, `/summary`, `/quit`.

**Python notes for the unfamiliar**
- `[0]` takes the first item of a list; `['text']` takes a value from a dictionary by key.
- `f"{x:.2f}"` formats a number to 2 decimals.
- `a @ b` is matrix multiplication in NumPy.
- `.venv` is an isolated folder of installed packages for this project.

### The web interface, `app.py`

A thin layer on top of the same functions: it adds no new retrieval or prompting logic.
- `respond(message, history, selected)` runs one turn: picks the checked indexes, converts the chat history to plain
  role/text messages, calls `assistant.answer`, and returns the updated chat and a sources panel.
- `summarize_file(name)` calls `assistant.summarize` for the chosen opinion.
- Errors (for example expired AWS credentials) are shown in the chat instead of crashing the page.
- Gotcha found while testing: Gradio 6 returns each past message's content as a list of blocks
  (`[{'text': ..., 'type': 'text'}]`) and not as a string. A first version silently dropped the conversation memory
  because of that; `message_text()` handles both forms. Lesson: test the integration seam, not only your own functions.

---

## 5. A real example, traced

Question: *"What did the Court decide about the standing of a tester who sues hotels over website accessibility?"*

1. Embed the question -> 512 numbers.
2. Search found 5 chunks, all from `601us1r01_q86b.pdf` (pages 4, 6, 7, 13, 14), the case *Acheson Hotels v. Laufer*.
3. The model answered: the Court did not decide standing because the case became moot after the plaintiff voluntarily
   dismissed her claim; there was a circuit split; Justice Thomas concurred and would have decided standing.
   Each statement carried `[n]` pointing at those pages.
4. Follow-up *"Who wrote the opinion, and was there a dissent?"* was rewritten to name the case, retrieved page 4
   ("Justice Barrett delivered the opinion of the Court"), and answered: Barrett wrote the majority; Thomas
   concurred in the judgment; the sources don't say whether there was a dissent.

---

## 6. Design decisions and trade-offs (expect "why did you...?")

| Decision | Why | Trade-off |
|---|---|---|
| Titan Text Embeddings v2 | On-demand on Bedrock, cheap, 512-dim option, no extra access form | A newer/stronger embedder might retrieve better |
| 1,200 chars, 200 overlap | A page is ~3,000 chars; ~2-3 chunks per page keeps ideas together | Too small loses context; too large dilutes the match |
| Chunks stay within a page | Exact citations | A passage spanning two pages is split |
| NumPy dot product, no vector DB | ~11k vectors search instantly; zero infrastructure | Doesn't scale to millions; no filtering |
| Two indexes (opinions, transcripts) | Different document types, can be searched alone or together | Merging results by raw score assumes comparable score scales |
| Sonnet for answers, Haiku for chores | Quality where it matters, low cost elsewhere | Two models to manage |
| Strict grounding prompt | Reduces invented facts | Model may answer "not in sources" when it is in the corpus but wasn't retrieved |
| `data/` and `indexes/` not committed | Large and reproducible | A reviewer must run download + ingest (or you hand them the indexes) |

---

## 7. How it was built with AI (for the "using the AI is the test" part)

You can describe your workflow like this:

1. **Set up the environment with an AI agent**: AWS CLI skills and the AWS MCP server, `aws login` credentials,
   and fixes for real errors (missing model-access form, `botocore[crt]`, region).
2. **Read the requirements** from the Confluence workbook with the agent and decomposed the project into five
   stages: documents, chunking, embeddings, retrieval + answer, conversation + summaries.
3. **Decided with the agent** on models (checked which embedding models allow on-demand use) and data sources
   (public Supreme Court documents only; no customer data).
4. **Built incrementally and tested at each stage**: one embedding call, then the index, then real questions.
5. **Verified output against source**: the first follow-up answer said "per curiam"; the PDF page said Justice
   Barrett wrote the opinion. That was caught by checking, then fixed in the prompt, and re-tested.
6. **Debugged from evidence**: the `KeyError: 'text'` was traced to a Claude 5.x model spending its token budget
   on hidden reasoning; fixed by switching the cheap model and reading only text blocks.

Key message: *the AI writes and explains code, but a human decides requirements, checks facts against sources,
and reads the errors.*

---

## 8. Things that went wrong (and what each taught)

1. **No model access** -> Claude on Bedrock needs a one-time use-case form; the console and CLI can lag a few minutes.
2. **Reasoning models and small token budgets** -> a model that reasons first can exhaust `maxTokens` and return no
   text. Always read only text blocks, and size budgets accordingly.
3. **Hallucinated "per curiam"** -> retrieval missed the page naming the author and the model guessed. Lesson:
   retrieval quality caps answer quality; prompt rules reduce but don't remove the risk. Verify against citations.

---

## 9. Likely questions, with short answers

1. **What is RAG and why use it here?** Retrieve relevant passages, then generate from them. Avoids giant prompts, cuts cost, enables citations.
2. **What is an embedding?** Numbers capturing meaning; similar texts give nearby vectors.
3. **How do you find the closest chunk?** Cosine similarity: a dot product between normalized vectors, top 5.
4. **Why overlap?** So an idea cut at a chunk boundary appears whole in a neighbor.
5. **Where do embeddings happen?** In Amazon Bedrock (Titan model), called once per chunk at ingest and once per question at query time.
6. **Why two indexes?** Different document types; users can scope the search; mirrors real systems with separate collections.
7. **How do you make follow-ups work?** Send recent history to the model, and rewrite the follow-up into a standalone search query before retrieval.
8. **How do you prevent made-up answers?** Grounding prompt, explicit "say so if missing", citations to verify, and testing; it's reduced, not eliminated.
9. **How do you know the citations are right?** Chunks carry file and page from ingestion; you can open that page and check (this caught a real error).
10. **How would you scale it?** Managed vector store (OpenSearch Serverless, pgvector, or Bedrock Knowledge Bases), hybrid search, re-ranking, OCR, caching.
11. **How would you evaluate quality?** A test set of questions with expected source pages; measure retrieval hit rate and answer correctness.
12. **What are the cost drivers?** Embedding tokens at ingest (one-time, small) and per question: embedding + rewrite + the answer model's input/output tokens. Check current Bedrock pricing.
13. **Security considerations?** Public data only; no credentials in code (short-lived `aws login` credentials); least-privilege IAM in production; don't log sensitive questions.

---

## 10. Hands-on exercises (to truly learn it)

1. Change `chunk_text` size to 600 and re-run ingest. Do answers get better or worse? Why?
2. Change `retrieve(k=5)` to 3 and to 10. What happens to answer completeness and cost?
3. Ask a question whose answer is not in the corpus ("What is the capital of France?"). Does it refuse correctly?
4. Print the retrieved chunks for a question and read them. Would *you* have answered from them?
5. Use `/use transcripts` and ask the same question as with `/use opinions`. How do the answers differ in style?
