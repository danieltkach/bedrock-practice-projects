# Legal Assistant with RAG (Amazon Bedrock)

A command-line research assistant that answers questions about real U.S. Supreme Court documents. It
retrieves the relevant passages first, then asks Claude to answer **only from those passages** and to
cite the exact file and page for every claim.

> Research aid only. It does not provide legal advice, and answers must be checked against the cited pages.

## Features

| Feature | How |
|---|---|
| PDF ingestion | `pypdf` extracts text page by page, keeping page numbers |
| RAG with multiple indexes | Two independent indexes: `opinions` (case law) and `transcripts` (oral arguments). Search all or one |
| Conversational interface | Keeps the last 6 messages; rewrites follow-ups ("and the dissent?") into standalone search queries |
| Precise citations | Each answer cites `[n]`; a source list gives file name, page, and index |
| Executive summaries | `/summary <file name part>`: summarizes each section, then merges them (map-reduce) |
| Web interface | Gradio app (`./run.sh web`): chat with index selector, source panel, and an executive-summary tab |
| Scale | 100+ documents per index (see "Dataset") |

## Architecture

```
 INGEST (once)                                   CHAT (every question)
 PDFs --pypdf--> pages --chunk--> chunks         question
                                   |                |  (follow-up? rewrite with Claude Haiku 4.5)
                          Titan Embeddings v2       v
                                   |             embed question (Titan v2)
                                   v                |
                      indexes/<name>.npy (vectors)  v
                      indexes/<name>.json (text,   cosine search -> top 5 chunks (across active indexes)
                                file, page)         |
                                                    v
                                         prompt = numbered sources + question
                                                    |
                                          Claude Sonnet 4.5 (answers ONLY from sources)
                                                    |
                                                    v
                                         answer with [n] citations + source list
```

## Quick start

Prerequisites: Python 3.10+, AWS CLI v2.32+, an AWS account with Bedrock access in `us-east-1`
(Claude models need the one-time Anthropic use-case form, which the Bedrock console shows the first
time you open a Claude model in the playground).

```bash
aws login --remote --region us-east-1   # or any other way to get AWS credentials
./run.sh setup       # virtualenv + dependencies
./run.sh download    # ~250 public PDFs into data/ (about 120 MB)
./run.sh ingest      # builds both indexes (a few minutes; Bedrock embedding calls)
./run.sh chat        # start the assistant in the terminal
./run.sh web         # or the web interface at http://127.0.0.1:7860
```

Or everything in one go: `./run.sh all`.

### Web interface

`./run.sh web` starts a local Gradio app at http://127.0.0.1:7860 (not exposed to the network). The **Ask questions** tab
has checkboxes to choose which indexes to search, the conversation, and a panel listing the sources of the last answer.
The **Executive summary** tab summarizes any downloaded opinion. It uses exactly the same code as the terminal assistant
(`assistant.py`), so both behave identically.

### Using the terminal assistant

```
you> What did the Court decide about standing for a tester who sues hotels over website accessibility?
you> Who wrote the opinion, and was there a dissent?        <- follow-up, remembered
you> /use transcripts                                       <- search only the oral arguments
you> /use all
you> /indexes
you> /summary 601us1r01                                     <- executive summary of one opinion
you> /quit
```

## Project layout

```
rag.py                    shared logic: PDF reading, chunking, embeddings, vector search
ingest.py                 builds one index from a folder of PDFs
assistant.py              chat loop, retrieval, prompts, citations, summaries
app.py                    Gradio web interface (reuses assistant.py)
download_opinions.py      fetches opinions from supremecourt.gov
download_transcripts.py   fetches argument transcripts from supremecourt.gov
run.sh                    single entry point (setup / download / ingest / chat)
requirements.txt          pinned dependencies
STUDY_GUIDE.md            concepts, code walkthrough, and rehearsal Q&A
data/                     downloaded PDFs (not committed; regenerate with ./run.sh download)
indexes/                  generated vectors + metadata (not committed; regenerate with ./run.sh ingest)
```

## Configuration

All knobs are constants at the top of `rag.py` and `assistant.py`:

| Setting | Default | Meaning |
|---|---|---|
| `EMBED_MODEL` | `amazon.titan-embed-text-v2:0` | embedding model |
| `EMBED_DIMS` | 512 | vector size (smaller = less storage, slightly less precise) |
| `chunk_text(size, overlap)` | 1200 / 200 chars | passage size and shared edge between neighbors |
| `retrieve(k)` | 5 | passages sent to the model per question |
| `ANSWER_MODEL` | Claude Sonnet 4.5 | writes answers and executive summaries |
| `FAST_MODEL` | Claude Haiku 4.5 | cheap steps: query rewriting, per-section summaries |
| `HISTORY_TURNS` | 6 | past messages included |

## Design decisions

- **Page-level provenance.** Chunks never cross a page boundary, so every citation maps to one real page.
- **Overlap.** A sentence cut at a chunk edge still appears whole in the neighboring chunk.
- **Normalized vectors + NumPy.** A dot product equals cosine similarity. About 11k vectors per index search in
  milliseconds, so no vector database is needed at this scale.
- **Grounded prompt.** The system prompt forbids answering outside the sources and forbids inferring unstated facts.
- **Cheap model for cheap work.** Haiku handles rewriting and section notes; Sonnet handles final answers.
- **Public data only.** No customer data or credentials are involved.

## Known limitations and suggested improvements

| Limitation seen in testing | Improvement |
|---|---|
| Search can miss a page that holds a key fact (e.g. the author line of an opinion); the model then says the sources don't show it | Hybrid search: combine keyword (BM25) with embeddings; add a per-document header chunk |
| Out-of-scope questions are refused correctly, but the unrelated retrieved sources are still listed | Add a minimum similarity score; below it, skip the model call and say "not in the corpus" |
| Top-5 by similarity can return several near-duplicate chunks | Re-rank, or use maximal marginal relevance for diversity |
| Index lives in two flat files and is loaded fully into memory | Move to OpenSearch Serverless, pgvector, or Bedrock Knowledge Bases for larger corpora |
| Scanned PDFs without a text layer produce no text | Add OCR (e.g. Amazon Textract) |
| No automated quality checks | Build a small question/expected-page test set and measure retrieval hit rate |
| The web interface is single-user and local | Authentication, per-user conversation storage, and deployment behind a managed host |
| Credentials expire (`aws login` lasts up to 12 h) | Use an IAM role when deploying; add retry/backoff for throttling |
| Runs on one machine | Containerize (Docker) with credentials supplied by environment/role, once deployed beyond a laptop |

## Troubleshooting (all hit during development)

| Symptom | Cause and fix |
|---|---|
| `Model use case details have not been submitted` | Open any Claude model in the Bedrock playground once and complete the form; allow a few minutes |
| `Using the login credential provider requires ... botocore[crt]` | `pip install "botocore[crt]"` (already in `requirements.txt`) |
| `NoRegionError` | `aws configure set region us-east-1` or `export AWS_DEFAULT_REGION=us-east-1` |
| `Your session has expired` | `aws login --remote --region us-east-1` |
| Web page shows "Sorry, the request failed: ... reauthenticate" | The AWS session expired; run `aws login --remote --region us-east-1` and ask again (no restart needed) |
| `KeyError: 'text'` from a Claude 5.x model | Those models think first and may spend a small `maxTokens` budget on reasoning; use a model that doesn't (Haiku 4.5) or raise the budget |

## Dataset

Public documents from supremecourt.gov, fetched by the download scripts (polite 1 s delay between requests).

| Index | Content | Documents | Pages | Chunks |
|---|---|---|---|---|
| `opinions` | Slip opinions, terms 2023 and 2024 | 127 | 5,028 | 11,222 |
| `transcripts` | Oral argument transcripts, terms 2023 and 2024 | 123 | 13,326 | 19,593 |
