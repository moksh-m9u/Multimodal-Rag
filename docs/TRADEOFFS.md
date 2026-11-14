# Engineering trade-offs

This document records, for each component of the system, what we use today, what else we could use, and why. Every choice here should be one we can defend. Each section ends with a short "how to defend it" answer and the current status.

Scope is the current phase: a user uploads a datasheet-style PDF and asks questions about it. The future PCB design agent is deliberately left out.

**How to read the option tables.** Every alternative stays in the table, because the comparison is what makes a choice defensible. A marker in brackets after the option name records the owner's position:

- **(chosen)**: the owner's current choice. Work proceeds with it.
- **(future option)**: kept deliberately for later; not being built now.
- **(to research)**: an open to-do that needs research before a decision.
- **(not pursued)**: considered and set aside, with the reason in the section.

A chosen option is not frozen. When an agent works on that area it should still give its own opinion and ask the owner if it sees a better path, then carry on with the chosen option unless the owner changes it. Owner decisions recorded on 2026-11-13 are marked "Owner (2026-11-13)".

## The system in one paragraph

Ingestion is offline. A PDF is partitioned into elements (text, tables, images) by `unstructured`. The elements are grouped into chunks by section title, and a vision LLM writes one searchable summary per chunk. Only the summary is embedded (IBM watsonx Granite, 768 dimensions) and stored in Chroma. The original text, the tables (as HTML), and the image paths ride along as metadata.

At query time the question is embedded and Chroma returns 10 diverse chunks (MMR over 20 candidates). The original text, tables, and images are rebuilt from metadata and sent to Gemini as a multimodal prompt. FastAPI serves this as `/retrieve`, `/query`, `/query/stream` (SSE), and `/query/custom`, and Streamlit is the UI.

The central design idea is **index the summary, answer from the original**. Summaries are good for matching questions. Originals are good for exact numbers.

## 1. PDF extraction

**Today:** `unstructured.partition_pdf` with `strategy="hi_res"`, `infer_table_structure=True`, and image blocks extracted as base64 (`src/ingestion/extract.py`).

| Option | Strengths | Weaknesses |
|---|---|---|
| unstructured hi_res (current, chosen) | Layout model finds tables and figures; HTML tables; one library | Slow, heavy system dependencies (poppler, tesseract), a CPU-bound model on every page, table quality varies |
| PyMuPDF / pdfplumber | Very fast, exact text from digital PDFs, coordinates for every span | No layout understanding; tables and figures need custom logic |
| Docling (IBM) | Good table structure, reading order, page and bounding box provenance, runs locally | Another heavy model stack; newer |
| Unstructured's hosted Parse API (future option, the one planned for production) | Best table quality with no ops; same vendor and element model as the local parser, so one adapter covers both; `UNSTRUCTURED_API_KEY` and `UNSTRUCTURED_URL` are already in `.env`, and the comparison below was measured with them | Cost per page, data leaves the machine (datasheets under NDA), returns image boxes but no image bytes, vendor lock-in |
| Other hosted parsers (LlamaParse, Azure Document Intelligence, Reducto) | Strong table quality with no ops | Cost per page, data leaves the machine, a second vendor and element format to adapt, no key or account set up |
| Vision retrieval (ColPali, ColQwen) | Skips OCR; retrieves whole page images; strong on figure-heavy pages | About 1,000 vectors per page, needs a GPU-class model, gives no text to quote or compute with |

Datasheets are mostly digital (not scanned) PDFs, so text extraction is easy and tables are the hard part. The biggest gap in the current choice is not the parser, it is **provenance**: we do not keep the page number and bounding box of each chunk, so answers cannot cite "page 12, Table 6". Docling or PyMuPDF both give coordinates.

**How to defend it:** "I needed text, tables, and figures from one tool without building a layout pipeline myself, so I used unstructured hi_res. Its cost is speed, which does not matter for offline ingestion. The thing I would change first is keeping page numbers and coordinates, because citations are what make a datasheet assistant trustworthy."

**Owner (2026-11-13):** local unstructured hi_res in development, Unstructured's hosted Parse API in production. The reason is deployment: hi_res needs poppler, tesseract, and a layout model on every page, and the free tiers of Render or Railway cannot run that within their memory and CPU limits. One environment variable, read in `config/settings.py`, picks the backend (planned name `PARSER_BACKEND=local|hosted`), so switching is an edit to the deployment's environment, not to code. For now the parser is the only setting that differs between development and production.

### Local hi_res compared with the hosted Parse API

Measured on 2026-11-13 with `data/AN699.pdf` (13 pages), local settings as in `config/settings.py`, hosted call `POST {UNSTRUCTURED_URL}/parse` with `output=elements`, `include=coordinates`, `include=table_html`, and the default `profile=balanced`.

| | Local hi_res | Hosted Parse (balanced) |
|---|---|---|
| Time | 162 s on the M4 (first run includes model loading) | 46 s |
| Elements | 276 | 223 |
| Tables with HTML | 4; three of them almost empty (1 of 30 cells filled) although the text was read | 5; 24 to 30 of 30 cells filled |
| Images detected | 21, saved as JPEG files | 48, as elements with coordinates only |
| Page number and coordinates | on every element | on every element (`PixelSpace`, page rendered at 2125 x 2750) |
| Image bytes | yes (files on disk) | **no**: the v2 API has no `image_base64` field and no image extraction setting |

What this means:

- Hosted table quality is clearly better on this document, which matters most for datasheets. One document is not proof; Checkpoint 5's evaluation set should confirm it.
- Element types differ (hosted has `Footer`, `PageNumber`, `Form`; local has `Header`, `Formula`, `UncategorizedText`), and the hosted metadata has fewer keys. Code that reads elements must depend only on the shared fields: `type`, `text`, `metadata.page_number`, `metadata.coordinates`, `metadata.text_as_html`. A small adapter can turn both outputs into one internal shape, so chunking does not care which backend ran.
- The hosted API does not return figure images. Because every image element has a page number and a bounding box, the figure can be cropped from a render of that page (for example with `pypdfium2`, already installed with unstructured). That keeps one way of producing figure files for both backends, and the same boxes later drive highlighted citations.
- `chunk_by_title` runs on our side on the elements, so chunking stays identical.
- The `/extract` endpoint is a different tool: it fills a JSON Schema you supply (for example part number, package, absolute maximum ratings). It does not replace Parse. It is a candidate for the key-specs card (`docs/PRODUCT_RESEARCH.md`, item 5), and can reuse a finished Parse by its id.
- The hosted API sends the document to Unstructured. That is acceptable for public datasheets and is the reason a local mode stays available for confidential ones.
- The `best` profile was not tested.

**Status:** keep hi_res locally. Add the backend switch, the element adapter, and figure cropping in Checkpoint 7, together with page provenance.

## 2. Chunking

**Today:** `chunk_by_title` with max 3,000 characters, a new chunk after 2,400, and chunks under 500 combined (`src/ingestion/chunk.py`, `config/settings.py`).

| Option | Strengths | Weaknesses |
|---|---|---|
| By title (current, chosen) | Follows the document's own sections ("Electrical Characteristics", "Layout Guidelines"); tables stay with their heading | Long sections get cut at arbitrary points; very uneven chunk sizes |
| Fixed size with overlap | Simple, predictable token counts | Splits tables and spec rows mid-way, which is fatal for datasheets |
| Page-level | Matches how engineers cite datasheets; pairs naturally with page images | A page mixes unrelated topics; weaker matching |
| Element-level plus parent retrieval | Small units for matching, the whole section for the answer | More moving parts to explain |

There is an interaction with embeddings: the watsonx model accepts only 512 tokens, roughly 2,000 characters. 45 of the 270 summaries (about 17 percent) are longer than 2,000 characters and are silently truncated. An earlier count of 39 of 312 included the TLV1117 chunks twice and left out LM555. Either chunks and summaries must be shorter, or the embedding model needs a longer context.

**How to defend it:** "Datasheets are organised by section, and a question usually targets one section, so title-based chunking matches the structure. Fixed-size chunking would split electrical characteristics tables. I measured that about 17 percent of summaries exceed the embedding limit, so the next step is capping summary length."

**Owner (2026-11-13):** chunk by title only, for now.

**Status:** keep. Cap summary length for new uploads and measure the effect with the evaluation set (Checkpoint 5).

## 3. Enrichment (chunk summaries)

**Today:** for every chunk, GLM-4.5V on the Hugging Face router receives the text, the tables, and the images and writes a summary (`src/ingestion/enrich.py`). The summary becomes the embedded `page_content`. The Hugging Face free credit this relied on no longer exists, so this step currently has no working provider.

Why summarise at all: a raw chunk might be a table of numbers that embeds poorly against a question like "what is the dropout voltage at 1 A". A summary written in words ("dropout voltage versus load current, 1.2 V typical at 800 mA") matches questions far better, and it describes figures that would otherwise be invisible to a text embedder.

| Option | Strengths | Weaknesses |
|---|---|---|
| Hosted VLM on HF router (GLM-4.5V, previous) (to research) | Strong vision, no local hardware | Free credit is gone; data leaves the machine; the provider can change or disappear |
| Gemini Flash-Lite (already used for generation) | Same key and vendor as generation; strong vision; cheap with a free tier; has an OpenAI-compatible endpoint, so `enrich.py` barely changes | Rate limits on the free tier; one vendor for two jobs |
| Local VLM on the M4 (Qwen2.5-VL-7B, Qwen3-VL-8B, Gemma 3 12B via Ollama, LM Studio, or mlx-vlm) (to research) | Free, private (works for NDA datasheets), reproducible, no credits to lose | Slower; a 16 GB machine limits the model to about 8B parameters at 4-bit; quality is somewhat lower on dense tables |
| No summary, embed raw text | Simplest; nothing to hallucinate | Figures are invisible; tables match badly |
| Multi-vector (embed both raw and summary) (to research) | Best recall | Twice the vectors; more to explain |

A summary can hallucinate. That is acceptable only because the answer is generated from the original content, not the summary. The summary's job is just to be found.

Switching provider is a configuration change: `ENHANCEMENT_BASE_URL` and `ENHANCEMENT_MODEL` point at any OpenAI-compatible server (Ollama serves one at `http://localhost:11434/v1`).

The summarizer is needed only for **new** documents. The existing 270 summaries are stored in Chroma and in `json/`, and they move to Postgres as they are (section 5).

**How to defend it:** "Summaries are an index, not a source. They translate tables and figures into the language people ask in, which raises recall, and the generator still reads the original so a bad summary costs recall, never correctness. Summaries are produced once and stored, so losing a provider never forces re-summarizing the corpus."

**Owner (2026-11-13):** keep the existing summaries exactly as they are (270, after the duplicate TLV1117 copy was removed on 2026-11-14; recorded as 312 at the time). For new uploads, three options go on the to-do list, each needing research before a decision:

1. **A hosted multimodal model** that accepts at least 5 to 6 images per request, so one chunk's figures fit in a single call. GLM-4.5V is the reference point; any comparable model through an OpenAI-compatible endpoint qualifies.
2. **A local vision language model.** Deep research is needed on both the serving framework (Ollama, LM Studio, mlx-vlm, llama.cpp) and the model (for example the Qwen-VL and Gemma families), judged on table reading, images per call, speed on the owner's 16 GB M4, and memory.
3. **Multi-vector embeddings**, embedding more than one representation per chunk (raw text and summary, or several vectors per chunk).

**Status:** to research before Checkpoint 7, which needs a summarizer for uploads.

## 4. Embeddings

**Today:** `ibm/granite-embedding-278m-multilingual` on watsonx.ai (eu-de), 768 dimensions, 512-token input with truncation (`src/embed.py`). It replaced a local 384-dimension Hugging Face model.

| Option | Strengths | Weaknesses |
|---|---|---|
| watsonx Granite 278M (current, chosen) | No model in the API image; small and fast; multilingual | 512-token limit; a network call per query; a cloud account dependency |
| Local sentence-transformers (bge-small, e5) (not pursued) | No network, free | Brings torch into the image (about 2 GB); CPU latency |
| OpenAI text-embedding-3, Gemini embedding | Long context (8k tokens), strong quality | Cost; another vendor |
| Domain fine-tuned embedder | Best for part numbers and units | Needs training data we do not have yet |

Changing the embedding model always means re-embedding every chunk, because vectors from different models are not comparable (we saw "expecting 384, got 768"). That is why there is exactly one embedding factory in `src/embed.py`, used by both ingestion and retrieval. Re-embedding is cheap (one API call per chunk); re-summarizing is the expensive step, and the two are independent.

A known weakness of every dense embedder: part numbers (`LM2596S-5.0`) and exact parameter names are matched poorly. Hybrid search (section 6) addresses this, not a bigger model.

**How to defend it:** "Moving embeddings to a hosted API took torch out of the serving path, so the API container is small and starts fast. I accept a network call per query. The 512-token limit is the real constraint, and I handle it by capping summary length rather than buying a bigger model."

**Owner (2026-11-13):** keep watsonx Granite 278M. A local sentence-transformers model is not pursued, because it would put torch back into the API image.

**Status:** keep. Torch and transformers were removed from the image (Checkpoint 1, done).

## 5. Vector store

**Decided (2026-11-12): Postgres + pgvector**, replacing Chroma and also holding application data.

**Today:** Chroma, persisted as files in `dbv2/chroma_db`, cosine HNSW. It is baked into the Docker image at build time.

| Option | Strengths | Weaknesses |
|---|---|---|
| Chroma embedded (current) | Zero ops; one directory; perfect for a read-only demo corpus | No concurrent writers; the store lives inside the container, so uploads are lost on redeploy; no joins with application data |
| Chroma on a Railway volume | Uploads survive redeploys | Volumes force one replica and cause downtime on redeploy; still no relational data |
| Postgres + pgvector (chosen) | One database for vectors and application data; SQL joins ("chunks of documents this user uploaded"); transactions; full-text search for hybrid retrieval; HNSW up to 2,000 dimensions; managed backups | You manage indexes and migrations; slightly more setup |
| Qdrant, Weaviate, Pinecone | Built for vectors: filtering, quantization, scale | A second database next to Postgres anyway; vendor or ops cost |

The deciding factor is the upload feature. Once users upload documents, we need document records, job status, chat history, the cache, and later users. That is relational data, and it must live in a real database anyway. With pgvector the chunks live in the same database, so deleting a document deletes its vectors in one transaction, and hybrid search is a single SQL query. At our scale (thousands of chunks, not hundreds of millions) pgvector's performance is not a concern.

### Migrating without rebuilding

Each Chroma record already holds everything a pgvector row needs:

| Chroma field | Postgres column |
|---|---|
| `id` | `chunks.id` |
| `document` (the summary) | `chunks.summary` |
| `embedding` (768 floats) | `chunks.embedding vector(768)` |
| `metadata.raw_text`, `metadata.tables_html` | `chunks.raw_text`, `chunks.tables_html` |
| `metadata.image_paths` | one row per path in `images` |
| `metadata.chunk_id`, `has_table`, `has_image` | `chunks` columns (the flags can become computed) |
| `metadata.source` | `documents.name`, referenced by `chunks.document_id` |

So the migration is a one-off copy: `collection.get(include=["documents", "embeddings", "metadatas"])` and a bulk insert. It needs **no summarizer and no embedding calls**, because the embedding model stays the same. Figure JPEGs stay on disk under `dbv2/images` and are referenced by path.

`json/` is a second, independent copy of the summaries. If the Chroma files were ever lost, Postgres could be rebuilt from `json/` with only embedding calls (cheap), still without a summarizer.

What would force real re-work:

- **Changing the embedding model** means re-embedding every chunk. That costs only embedding calls.
- **Changing chunk boundaries** (a different chunk size, or page-level chunks) means new chunks, so new summaries. This is the only case that needs the summarizer for the existing corpus.
- **Adding page numbers and bounding boxes** to existing chunks needs re-extraction with `unstructured`. If chunk boundaries stay the same, the old summaries can be reused.

Verification: run the evaluation set against Chroma and against pgvector. Identical vectors and the same distance metric (cosine) should give the same top-k, apart from approximate-search tie-breaks.

**How to defend it:** "Chroma was right for a fixed corpus baked into an image. Uploads turn this into a multi-document, stateful system, and keeping vectors next to the document records in Postgres removes a whole class of consistency bugs. Because summaries and embeddings are stored artifacts, the migration was a data copy, not a rebuild, and I proved it with the evaluation set."

**Status:** Checkpoint 6. The host is still to decide.

## 6. Retrieval

**Today:** MMR with `k=10` and `fetch_k=20` (`src/retrieval/search.py`). The prompt then uses the text of only the first 5 chunks but the images of all 10, which is inconsistent and is a bug to fix.

| Option | Strengths | Weaknesses |
|---|---|---|
| Plain similarity top-k | Simplest; the most relevant chunks | Near-duplicates (the same table on two pages) crowd out other evidence |
| MMR (current) | Diversity across sections | Can push out the second-best relevant chunk; the lambda setting is a magic number |
| Hybrid (Postgres full-text plus dense, fused with RRF) (chosen, after the migration) | Exact matches on part numbers, pin names, and parameter symbols | Tuning the fusion; free once on Postgres |
| Cross-encoder reranker on the top 20 to 30 | Large precision gains | An extra model call per query |
| Metadata filtering by document or document type (to research) | Required once users have their own documents | Needs a document ID on every chunk |

pgvector has no built-in MMR. Fetch the 20 nearest rows in SQL and apply MMR in Python. It is about 15 lines of code and easy to explain.

For datasheets, hybrid search is the most valuable addition, because questions contain literal tokens ("VREF", "TLV1117-33", "θJA") that dense embeddings blur.

**How to defend it:** "MMR avoids sending the model five copies of the same table. The next improvement is hybrid search, because engineers search with exact symbols, and I would justify it with the evaluation set rather than intuition."

**Owner (2026-11-13):** go hybrid once on Postgres, since Postgres gives full-text search for free. Metadata filtering by document and by document type is wanted, but the design (which fields, who sets the type, how the UI exposes it) still needs thought.

**Status:**
- Fix the 5-versus-10 inconsistency (Checkpoint 3).
- Add document filtering (Checkpoint 7).
- Add hybrid search after the migration.

## 7. Generation

**Today:** Gemini 3.5 Flash-Lite at temperature 0 through `langchain-google-genai`. The prompt contains chunk text, tables as HTML, and images as base64 `image_url` blocks (`src/retrieval/generate.py`). `/query/custom` lets the caller pick another provider and model with their own API key (`config/providers.py`).

| Option | Strengths | Weaknesses |
|---|---|---|
| Gemini Flash-Lite (current, chosen) | Cheap, fast, native image input, long context | Weaker multi-step reasoning on electrical calculations |
| Larger models (Gemini Pro, Claude, GPT) | Better reasoning and calculation | Cost and latency |
| Local VLM | Private | Slow on a laptop; weak at long multimodal prompts |
| Calculations in code (tool calls) | Exact arithmetic with units | Needs a tool loop; belongs to a later phase |

Images are expensive in tokens. `max_images` exists to cap them. Sending figures only when the question looks visual ("graph", "curve", "figure", "pinout") is a cheap improvement.

The custom-provider path duplicates the default path today. Both become one function that receives an LLM object (see section 8).

**How to defend it:** "Temperature 0 because this is lookup, not creativity. A small multimodal model because most questions are retrieval-bound: if the right table is in the prompt, a small model reads it correctly. The provider is injectable so the same pipeline can be evaluated against several models."

**Owner (2026-11-13):** Gemini Flash-Lite works well enough; do not change the default model now.

**Status:** merge the two paths (Checkpoint 2).

## 8. API layer

**Today:**
- One FastAPI app in `api/main.py` with seven endpoints.
- `RAGService` loads the store once and is shared through `app.state`.
- Synchronous LangChain calls run in `asyncio.to_thread`.
- Figures are served from an in-memory `ImageRegistry` under `/images/{response_id}/{index}`, and responses also carry every figure as base64.
- Errors often return 200 with a "Sorry..." answer.

**Decided (2026-11-12):**

| Change | Why | Cost |
|---|---|---|
| Routers in `api/routers/` | Each file explains one area; `main.py` reads as a table of contents | None; paths stay the same |
| Remove `images_base64` from responses | Responses shrink by megabytes; the UI already loads images by URL | Breaking for clients that read base64 |
| Stable image URLs `/v1/images/{record_id}/{position}`, served straight from disk | Cacheable, survive restarts, work across replicas; deletes `ImageRegistry` and its memory growth | Breaking for clients that stored old URLs |
| One `POST /query` with an optional `llm: {provider, model, api_key}` object, replacing `/query` and `/query/custom` | One code path to explain and test; the default LLM is used when `llm` is absent | Breaking for `/query/custom` callers |
| `POST /query/stream` with the same body as `/query` | Long questions and options do not belong in a URL (logs, length limits); one schema for both | Browser `EventSource` supports only GET, so clients read the stream with `fetch` or `httpx` |
| Real HTTP errors: 422 for bad input, 404 for unknown images or models, 502 when the LLM provider fails, 503 when the store is not ready | Clients and monitoring can tell failure from an answer; the UI can show the reason | The UI must handle error codes |

The API key in the `llm` object travels in the request body over HTTPS. It must never be logged or stored, and it must never appear in traces.

**Decided (2026-11-13): a `/v1` prefix** on every endpoint except `/` and `/health`. It lets the API change later (`/v2`) without breaking clients, at the cost of slightly longer URLs. Health stays unversioned because uptime probes and Railway's health check should not break when the API version changes.

### Planned endpoints

| Endpoint | Replaces | Checkpoint | What it does |
|---|---|---|---|
| `GET /` | same | 2 | Index of endpoints and a link to `/docs` |
| `GET /health` | same | 2 | Store loaded or not; 503 while loading |
| `POST /v1/retrieve` | `POST /retrieve` | 2 | Chunks only, no answer |
| `POST /v1/query` | `POST /query` and `POST /query/custom` | 2 | Chunks plus an answer. An optional `llm` object picks the model; without it the default Gemini model is used |
| `POST /v1/query/stream` | `GET /query/stream` | 2 | Same body as `/v1/query`, answer streamed over SSE |
| `GET /v1/images/{record_id}/{position}` | `GET /images/{response_id}/{index}` | 2 | One figure as JPEG, read from disk |
| `GET /v1/providers` | the UI importing `config/providers.py` | 4 | Providers and their models, for the model picker |
| `GET /v1/documents` | the UI reading `json/` | 4 | Indexed documents with chunk counts |
| `GET /v1/documents/{document_id}/chunks` | the UI reading `json/` | 4 | Every chunk of one document, for the chunk explorer |
| `POST /v1/documents` | new | 7 | Upload a PDF; returns 202 with a document id and status `queued` |
| `GET /v1/documents/{document_id}` | new | 7 | Ingestion status (`queued`, `processing`, `ready`, `failed`), pages, chunk count |
| `DELETE /v1/documents/{document_id}` | new | 7 | Remove a document with its chunks and images |
| `POST /v1/queries/{query_id}/feedback` | new | 6 | Thumbs up or down plus a comment on one answer |

Request body shared by `/v1/query` and `/v1/query/stream`:

```json
{
  "query": "What is the dropout voltage of the TLV1117 at 800 mA?",
  "top_k": 10,
  "max_images": 4,
  "document_ids": ["tlv1117"],
  "llm": {"provider": "groq", "model": "openai/gpt-oss-20b", "api_key": "...", "temperature": 0, "max_tokens": 1024}
}
```

`document_ids` arrives in Checkpoint 7; until then all documents are searched.

Response changes:
- **Removed:** `images_base64` and `response_id`.
- **Added, as the features arrive:**
  - `query_id` (Checkpoint 6), to attach feedback;
  - `cached` (Checkpoint 8);
  - `score`, `document`, and `page` on each chunk;
  - `image_urls` built from `record_id`.

`record_id` is the store's record id, a UUID, so it is globally unique. Neither `chunk_id` can key a figure: the stored one restarts at 1 in every document (78 distinct values across 270 records), and the one in the response is a 1-based position within that response.

Stream events for `/v1/query/stream`:

| Event | Data |
|---|---|
| `retrieval` | The chunks, as in `/v1/retrieve` |
| `token` | One piece of the answer |
| `done` | Usage, latency, and cost |
| `error` | Status and message, sent instead of `done` when the provider fails mid-stream |

Errors are returned as JSON with a `detail` field:

| Status | When |
|---|---|
| 404 | An unknown image, document, provider, or model |
| 413 | An upload that is too large |
| 422 | Invalid input |
| 502 | The LLM provider failed or rejected the key |
| 503 | The store is not ready yet |
| 504 | The provider timed out |

How FastAPI fits with LangChain: build the LLM, retriever, and later any LangGraph graph once in the lifespan handler and store them on `app.state`. In handlers, `await chain.ainvoke(...)` for a full answer and `async for chunk in chain.astream(...)` wrapped in a `StreamingResponse` for SSE. Synchronous-only code goes through `asyncio.to_thread` so one slow request does not block the others.

**How to defend it:** "Handlers are thin and the service owns the expensive objects, so the code is testable without HTTP and the event loop never blocks. I changed the API shape only where the old shape caused real problems: base64 bloat, image URLs that died with the process, two endpoints doing the same work, and failures disguised as answers."

**Owner (2026-11-13):** keep the planned endpoints as proposed. The owner may review them with another agent before Checkpoint 2 starts.

**Status:** Checkpoint 2.

## 9. Frontend

**Today:** Streamlit, `app/dashboard.py` at about 2,100 lines with four pages. It talks to the API through `app/api_client.py`, but still imports `config` and `src.logger` and reads `json/` directly.

| Option | Strengths | Weaknesses |
|---|---|---|
| Streamlit (current) | Python only; fast to build inspection tools | Reruns the script on every interaction; awkward for streaming chat and uploads at scale; hard to make look like a product |
| React with TypeScript (chosen, later phase; Next.js or Vite still to decide) | A real product UI; proper streaming and file upload | A second language and build system |
| Gradio | Very quick chat and upload demos | Less control |

For this phase Streamlit is right: the goal is an explainable backend, and the UI is a client of the API. Strict separation (HTTP only) is what makes swapping the UI later cheap.

**How to defend it:** "The UI is deliberately a thin HTTP client, so the backend is the product and the UI is replaceable. Streamlit let me build inspection tools for chunks and retrieval quickly, which matters more than polish at this stage."

**Owner (2026-11-13):** the product UI will be a React and TypeScript app in a later phase. Next.js or a plain Vite single-page app is decided when that phase starts. Streamlit stays as the internal inspection tool.

**Status:** separate and split (Checkpoint 4).

## 10. Main database

**Decided (2026-11-12): Postgres + pgvector**, one database for everything.

**Today:** none. There is nowhere to record documents, ingestion jobs, chats, queries, answers, or feedback.

Planned schema:

```
documents  (id, name, filename, sha256, pages, status, created_at)
chunks     (id, document_id, page, bbox, raw_text, tables_html, summary, embedding vector(768))
images     (id, chunk_id, position, path)
chats      (id, created_at)                      -- user_id added when auth arrives
queries    (id, chat_id, question, model, top_k, created_at, latency_ms, tokens, cost, cached)
retrievals (query_id, chunk_id, rank, score)
answers    (query_id, text, error)
feedback   (query_id, rating, comment)
cache      (id, question, question_embedding vector(768), part_numbers, model, top_k,
            corpus_version, answer, created_at)
```

The `queries`, `retrievals`, and `answers` tables turn every real question into evaluation data and give the semantic cache something to read. A `users` table comes with auth in a later phase.

Hosting options:

| Option | Strengths | Weaknesses |
|---|---|---|
| Railway Postgres (chosen, tentative) | Same platform as the API; private networking | pgvector must be enabled (use the pgvector template image); backups need setting up |
| Neon | Serverless Postgres, pgvector built in, point-in-time restore, branching (a throwaway copy of the database per test), generous free tier | Cold start of a second or so on the free tier after idle |
| Supabase | Postgres plus pgvector plus auth and file storage (useful for uploaded PDFs and later users) | More platform than we need now; free projects pause after a week idle |

Migrations with Alembic so the schema is versioned in git. Use SQLAlchemy Core or a thin repository module, not a heavy ORM, so queries stay readable.

**How to defend it:** "A RAG system without a database cannot answer basic product questions: which documents exist, what users asked, what failed. I added Postgres for that, and used pgvector so vectors live in the same transaction as the document they belong to."

**Owner (2026-11-13):** Railway Postgres, tentatively, because the API is planned for Railway too. Hosting as a whole is still open and will be discussed before Checkpoint 6.

**Status:** Checkpoint 6. Host tentatively Railway Postgres.

## 11. Semantic cache and router

**Idea:** before retrieval, embed the question and look for a previous question with cosine similarity above a threshold (start near 0.95). On a hit, return the stored answer. With pgvector the cache is just another table with a vector index.

Rules that make it safe for datasheets:

- **Part numbers must match exactly.** "Max current of LM317" and "max current of LM337" are semantically almost identical and have different answers. Extract part-number-like tokens and require equality.
- **The cache key includes the model, `top_k`, `max_images`, and a corpus version.** Uploading or re-indexing a document bumps the version and invalidates stale answers.
- **Never cache failures.**
- Return `cached: true` in the response so it is visible.

**Router model (optional):** a small classifier, or a cheap LLM call, decides whether a question needs retrieval, is a cache hit, or is a chit-chat or clarification question. It saves cost, but every misroute is a wrong answer, so it must be measured on the evaluation set before it is trusted.

**How to defend it:** "Repeated questions are common on a fixed document. The risk is false hits on near-identical part numbers, so the cache combines semantic similarity with exact identifier matching and is versioned by corpus."

**Status:** Checkpoint 8.

## 12. Observability

**Today:** LangSmith tracing on generation, with chunk summaries and images attached to runs (`src/tracing.py`). Logs go to stdout and `logs/pipeline.log`.

LangSmith is good for inspecting single requests. It does not replace an evaluation set or request logs in the database. Keep it, keep it optional (tracing off when no key is set), and stop writing debug files such as `last_prompt.txt` from the request path.

**Status:** keep. Clean up debug writes (Checkpoint 1).

## 13. Deployment

**Today:** a Docker image on Railway with the Chroma store copied in at build time. The Railway service is currently down. Streamlit Cloud hosts the UI.

Until 2026-11-13 the image also installed CPU torch, transformers, and the OpenCV system libraries. They were left over from the local embedding model. Embeddings are now a watsonx.ai API call and nothing in the backend imports any of them, so they were removed.

After the migration the API container becomes stateless: vectors and data live in Postgres, so the API can be rebuilt or scaled freely.

| Option for figure images and uploaded PDFs | Strengths | Weaknesses |
|---|---|---|
| Baked into the image (current, for the fixed corpus) | Simple, reproducible | New uploads cannot be stored |
| Railway volume | Uploads persist | Single replica; redeploy downtime |
| Object storage (S3, Cloudflare R2, Supabase Storage) | Stateless API; any number of replicas | One more service and credentials |

### Where figure images live: files, not database rows

The standard practice is to store **bytes in file or object storage and metadata in the database**. Images and PDFs are not stored inside Postgres.

- **The database** holds one `images` row per figure: `chunk_id`, `position`, `page`, `bbox`, width, height, and a `storage_key` (a path today, an object key later).
- **The bytes** live on disk now (`dbv2/images/`) and in object storage once uploads arrive (Checkpoint 7).

Why not `bytea` columns in Postgres:
- Every backup, restore, and replica carries the image bytes.
- Rows get large and slow down queries that never needed the image.
- Database storage costs far more per GB than object storage.
- The image cannot be served from a CDN.

Postgres can store binaries, and for a few hundred small JPEGs it would work. It just stops scaling at exactly the moment uploads start.

The URL is what stays stable, not the storage. `GET /v1/images/{record_id}/{position}` is the contract. Behind it:
- **Checkpoint 2:** the handler builds the file path from the chunk's stored `image_paths`.
- **Checkpoint 6:** it looks up `storage_key` in the `images` table.
- **Checkpoint 7:** it may redirect (307) to a short-lived signed object-storage URL.

Clients never notice any of these changes.

**Status:**
- Removed torch, transformers, and the OpenCV system libraries (Checkpoint 1, done).
- `.dockerignore` is an allowlist, so the image holds only `api/`, `src/`, `config/`, and `dbv2/`; 1.34 GB down to 1.17 GB (Checkpoint 1, done).
- Decide image and PDF storage with uploads (Checkpoint 7).
- Restore Railway (Checkpoint 9).

## 14. Testing and evaluation

**Today:** 29 pytest tests that mock external services, plus `evals/testing.json` and an evaluation dashboard.

Unit tests prove the code does what it says. They do not prove answers are right. The evaluation set does that: for each question, the expected document, the expected page or chunk, and the expected answer.

Metrics:
- retrieval hit rate at k;
- answer correctness (an exact value match for numeric specs, LLM-judged for prose);
- the rate of correct "not stated in the document" refusals.

The evaluation set is also what proves the Postgres migration changed nothing.

**How to defend it:** "Every retrieval change (chunk size, embedding model, hybrid search, cache threshold, a database swap) is a trade-off I can only claim with numbers, so I built the evaluation set before optimising."

**Status:** Checkpoint 5, before the migration.

## 15. Agent frameworks (LangGraph and others)

Out of scope for this phase. The current flow is a straight line (retrieve, then generate) and does not need a graph. LangGraph earns its place when there are loops and decisions: retrieve again if evidence is weak, call a calculator tool, ask a clarifying question, or check a schematic against several datasheets. When that time comes:

- the graph is compiled once in the FastAPI lifespan;
- it is invoked with `await graph.ainvoke(...)`;
- it is streamed with `graph.astream(...)` over SSE;
- its state is checkpointed in Postgres.

## Open decisions

1. Database host: tentatively Railway Postgres; hosting as a whole still to discuss (Checkpoint 6).
2. Summarizer for new uploads: research a hosted multimodal model, a local vision model, and multi-vector embeddings (section 3, Checkpoint 7).
3. Hosted Parse profile, `balanced` or `best`, measured on the evaluation set (section 1, Checkpoint 7).
4. Storage for uploaded PDFs and figure images: a volume or object storage (Checkpoint 7).
5. Metadata filtering by document type: which fields and how they are set (section 6).
