# AGENTS.md

Instructions for any coding agent (Claude Code, Codex, Cursor, and others) working in this repository. Read this file before making a change. The companion documents are:

- `docs/TRADEOFFS.md`: why each component is built the way it is, the alternatives, and the current recommendation.
- `docs/PRODUCT_RESEARCH.md`: what users who upload datasheets and PCB documents actually need, and where this project can stand out.
- `docs/GLOSSARY.md`: plain-language explanations of the technical terms used in the code and the docs.

## Why this repository exists

This is a multimodal RAG system over electronics documents (datasheets, application notes, and later schematics). A user uploads a PDF and asks questions about it. The answer draws on the text, the tables, and the figures in the document.

The long-term direction is an agent that assists PCB designers through the whole design process. That is explicitly out of scope for now. Do not add agent frameworks, MCP integrations, or web search until the current phase is finished.

**Current phase: make the codebase explainable.** The owner must be able to explain every module, every function, how data travels from PDF to answer, and defend every engineering decision. Much of the code grew quickly and is hard to follow. Every change in this phase should make the code smaller, clearer, or more correct. Behaviour stays the same unless a change fixes a real architectural mistake, and that fix is agreed first.

## How we work together

This is AI-assisted coding, not autonomous coding. The owner stays in control of every decision and every commit.

### The change loop

Follow these steps for every change, however small:

1. **Propose.** Say what you want to change, why, and which files it touches. If there is any decision to make (a library, an API shape, deleting something, a schema), ask first and wait for an answer. Do not pick silently.
2. **Implement one small change.** One concern per change. Prefer several small changes over one large one.
3. **Check it yourself.** Run the test suite and any direct check that applies (an import, a curl against the local API, a script). Report failures honestly with the output.
4. **Hand over.** Give the owner:
   - the exact commands to test the change, each in its own code block;
   - what they should see if it works;
   - a **navigation walkthrough**: where to start reading, the order of files and functions to follow, and the shape of the data at each step. Explain from first principles, assuming no prior context about the change;
   - anything not verified.
5. **Answer questions.** The owner will read the code and ask about it. Answer directly, quoting `file:line` locations.
6. **Commit only when asked.** When the owner says to commit, make one commit for that change.
7. **Push only when asked.** Pushes go straight to `main`, and only after the owner has tested and approved.

If something fails twice in a row, stop and explain the problem rather than looping on fixes. If you need an API key, a Docker image, a model download, or any new software installed, stop and explain exactly what it is, why it is needed, and how large it is, then wait.

### Navigation walkthrough format

After each change, write the walkthrough like this:

```
Start: api/routers/query.py:query()          the HTTP entry point
  -> api/service.py:RAGService.answer()      loads the store once, retrieves
  -> src/retrieval/search.py:retrieve_chunks()  returns list[Document]
  -> src/retrieval/generate.py:generate_answer() builds the prompt, calls the LLM
Data: QueryRequest -> list[Document] (metadata: raw_text, tables_html, image_paths) -> str answer -> QueryResponse
Try: set a breakpoint in X, or run the command below, and look at Y.
```

The owner wants depth here. At each hop, also explain from first principles what the library underneath does for us (FastAPI, LangChain, Chroma, unstructured, and so on): what it receives, what it returns, and what would break without it. Assume no prior knowledge of that library.

### Glossary

`docs/GLOSSARY.md` holds a short plain-language explanation of every technical term the project uses (SSE, HNSW, MMR, RRF, lifespan, and so on). When a change or a document introduces a term that is not there yet, add it in the same change and link to it rather than explaining it again elsewhere.

### Commits and branches

- The owner is the sole author. Never add a `Co-Authored-By` trailer or any other agent attribution, in commits or PR descriptions.
- Use Conventional Commits: `type(scope): summary` in the imperative mood, with a body that explains why. Types: `feat`, `fix`, `refactor`, `docs`, `test`, `perf`, `build`, `ci`, `chore`. Mark breaking API changes with `!` after the type and a `BREAKING CHANGE:` footer.
- Branch names describe the feature (`feat/<feature>`, `refactor/<area>`, `fix/<bug>`) and never contain the word "claude" or any agent name.
- Commit new documents and files promptly once the owner approves them. Untracked files cannot be recovered with git if they are deleted.

## Code style

- Every module starts with a docstring that says what the module is for and where it sits in the data flow.
- Every public function and class has a docstring: what it does, its arguments, what it returns, and what it raises. Private helpers get a one-line docstring.
- No banner comments (`# ----` or `# ====` dividers). If a file needs section dividers, it should probably be split into modules instead.
- No emojis anywhere: code, comments, logs, UI text, commit messages, and docs.
- Comments explain why, not what. Delete comments that restate the code.
- Use type hints on all function signatures.
- Keep things simple enough to explain. Prefer a plain function to a class, and a class to a framework. Do not add an abstraction until there are two real callers.
- Configuration lives in `config/settings.py` and is read from environment variables. No hardcoded secrets or URLs elsewhere.

## Architecture rules

- **Frontend and backend are strictly separate.** `app/` (Streamlit) talks to the backend only over HTTP through `app/api_client.py`. It must not import from `src/`, `api/`, or `config/` and must not read `json/` or `dbv2/` directly. Removing the remaining imports is part of the current phase.
- **One FastAPI app, many routers.** `api/main.py` only creates the app, sets up middleware and the lifespan, and includes routers. Each router file in `api/routers/` groups endpoints by purpose and stays thin: validate input, call a service function, shape the response.
- **Business logic lives in `src/`**, not in route handlers. `src/` must not import from `api/` or `app/`.
- **Heavy objects load once.** The embedding client, the vector store, and LLM clients are created at startup (FastAPI lifespan) or lazily once, never per request.
- **Blocking work stays off the event loop.** Synchronous LangChain calls inside async endpoints run through `asyncio.to_thread` or use the async LangChain methods (`ainvoke`, `astream`).
- **Errors are real HTTP errors.** Failures return 4xx/5xx with a JSON body, never a 200 with an apology string. A failing LLM provider returns 502.

## Repository map

```
config/settings.py      every tunable and environment variable
config/providers.py     catalogue of LLM providers and models for /query/custom
src/embed.py            the one embedding factory (IBM watsonx Granite, 768 dims)
src/ingestion/          offline pipeline: PDF -> elements -> chunks -> summaries -> Chroma
  extract.py              unstructured partition_pdf (hi_res, tables, images)
  chunk.py                chunk_by_title
  enrich.py               vision LLM summary per chunk (GLM-4.5V via HF router)
  export.py               chunks -> JSON export
  pipeline.py             runs the steps and writes Chroma
src/retrieval/
  search.py               load Chroma, build the MMR retriever, retrieve
  generate.py             build the multimodal prompt, call Gemini, stream, trace
src/documents.py        turns stored chunk metadata back into text, tables, images
src/tracing.py          LangSmith helpers
api/main.py             FastAPI app and endpoints (to be split into api/routers/)
api/service.py          RAGService: owns the store and retriever for the process
api/custom_service.py   /query/custom: same retrieval, user-chosen LLM
api/images.py           in-memory registry that serves figure JPEGs by URL
api/schemas.py          Pydantic request and response models
app/dashboard.py        Streamlit UI (2,100 lines, to be split into pages)
app/api_client.py       the only place the UI talks to the backend
app/rag_eval_dashboard.py  Streamlit viewer for evaluation results
scripts/                CLI entry points (ingest, query, Colab ingestion)
dbv2/                   the built Chroma store and extracted figure images
json/                   per-document chunk exports (summaries included) used to build dbv2
evals/                  evaluation questions and notes
tests/                  pytest suite (no network)
```

Data flow at a glance:

```
Ingestion (offline):  PDF -> partition_pdf -> chunk_by_title -> vision summary per chunk
                      -> embed summary (watsonx) -> Chroma (summary as page_content,
                      raw_text/tables_html/image_paths as metadata) + images on disk
Query (online):       question -> embed -> Chroma MMR (k=10 of 20) -> rebuild text,
                      tables, images from metadata -> multimodal prompt -> Gemini -> answer
```

## Commands

Use the project virtual environment at `/Users/moksh/Multimodal-Rag/.venv` (Python 3.13). A stale Homebrew `uvicorn` exists on the machine, so always call the venv binaries or activate the venv first.

```bash
source .venv/bin/activate
```

```bash
python -m pytest -q -p no:cacheprovider
```

```bash
uvicorn api.main:app --reload --port 8001
```

```bash
API_BASE_URL=http://127.0.0.1:8001 streamlit run app/dashboard.py
```

```bash
python -m scripts.ingest path/to/document.pdf
```

Notes:

- Tests must not call the network. Mock external clients (see `tests/test_embed.py`).
- Opening the Chroma store rewrites tracked files under `dbv2/chroma_db`. After any run that opens it, restore with `git checkout -- dbv2/chroma_db` unless the change is a deliberate re-index.
- Live checks load credentials from `.env` with `python-dotenv`. Never print, log, or read secret values. Refer to them only by variable name.
- Streamlit copies root-level keys from `.streamlit/secrets.toml` into `os.environ` and overwrites existing values. Keep that file out of git (it is ignored).

## Environment variables

`GEMINI_API_KEY`, `HF_TOKEN`, `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`, `API_BASE_URL`, `CORS_ALLOWED_ORIGINS`, `GENERATION_MAX_TOKENS`, and the LangSmith variables (`LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT`). See `config/settings.py` for defaults.

Also present in `.env` for the hosted parser: `UNSTRUCTURED_API_KEY` and `UNSTRUCTURED_URL` (the v2 Transform API). A planned `PARSER_BACKEND` (`local` in development, `hosted` in production) will choose between local hi_res and the hosted Parse API; see `docs/TRADEOFFS.md`, section 1.

## Decisions made

Recorded so no agent re-opens them. Dates are when the owner decided.

- 2026-11-12: **Postgres + pgvector** replaces Chroma and also holds application data (documents, chats, cache, and later users). Existing chunks, summaries, and embeddings are copied across; nothing is re-summarized.
- 2026-11-12: **API shape changes** (Checkpoint 2):
  - remove `images_base64` from responses;
  - serve figures from disk at stable URLs `/images/{chunk_id}/{position}`;
  - merge `/query` and `/query/custom` into one `POST /query` with an optional `llm: {provider, model, api_key}` object;
  - add `POST /query/stream` for streaming;
  - return real HTTP errors, with 502 when the LLM provider fails.
- 2026-11-13: **`/v1` prefix** on every API endpoint except `/` and `/health`, which stay unversioned for uptime probes. The planned endpoint list is in `docs/TRADEOFFS.md`, section 8.
- 2026-11-13: **Component choices.** `docs/TRADEOFFS.md` marks every option in its tables as (chosen), (future option), (to research), or (not pursued). Alternatives are never deleted. In short:
  - extraction: local unstructured hi_res in development, Unstructured's hosted Parse API in production, switched by one environment variable; free Render or Railway tiers cannot run hi_res;
  - chunking: by title only;
  - enrichment: keep the existing summaries; research a hosted multimodal model that takes 5 to 6 images per call, a local vision model (framework and model), and multi-vector embeddings;
  - embeddings: watsonx Granite 278M; no local sentence-transformers;
  - retrieval: hybrid search once on Postgres; metadata filtering by document and document type still to design;
  - generation: Gemini Flash-Lite stays the default;
  - frontend: React with TypeScript in a later phase; Next.js or Vite decided then;
  - database host: Railway Postgres, tentatively; hosting overall still to discuss.
- 2026-11-13: **First upload target is datasheets.** Privacy (local mode) is a later add-on. The demo user is a mix of student and working engineer.
- 2026-11-13: **Working style.** Keep the checkpoint order. There is no fixed deadline; understanding comes before speed. For the evaluation set, the agent drafts questions with expected answers and pages, and the owner reviews every one.
- A chosen option is not frozen. When working on an area, give your own opinion and ask if you see a better path, then proceed with the chosen option unless the owner changes it.
- Auth is deferred to a later phase.

## Checkpoints for the current phase

Each checkpoint is a set of small commits. Finish one, let the owner test and push it, then start the next. Items marked "decision" still need the owner's answer before work starts. Status lives here, so update it when a checkpoint is done.

**Checkpoint 0. Ground rules and docs.** Status: done.
`AGENTS.md`, `docs/TRADEOFFS.md`, `docs/PRODUCT_RESEARCH.md`.

**Checkpoint 1. Repository hygiene.** Status: in progress.
- Done: removed `chunks_export.json`, `rag_results.json`, `notebooks/raw.md`, `scripts/test_single_image_trace.py`, and the `last_prompt.txt` debug write. The owner chose to keep `parsed_pdf/`, `queries.txt`, and the notebook.
- Done: dropped `torch`, `transformers`, and the OpenCV system libraries from the API image; a local `docker build` and `/health` check passed on 2026-11-13. `docker run --env-file` does not strip quotes, so `.env` values must be unquoted.
- Next: the three items below, in order.
- Tighten `.dockerignore`. `COPY . .` currently puts `parsed_pdf/` (42 MB), `notebooks/`, `data/` PDFs, `assets/`, `.claude/`, and the JSON exports into the image, and the API reads none of them.
- Fix the data bug: `json/LM555TImer.json` is a copy of the TLV1117 export, so TLV1117 is indexed twice and LM555 is missing. The real LM555 chunks (36, summaries included) are in the root `chunks_huggingface.json`. The fix needs only embedding calls, no summarizer.
- Update the README to the current stack (watsonx, Gemini 3.5 Flash-Lite, k=10).

**Checkpoint 2. Backend structure.**
- Split `api/main.py` into `api/routers/system.py`, `retrieval.py`, `query.py`, `images.py`. `main.py` keeps only app creation, CORS, the lifespan, and `include_router` calls.
- Merge the default and custom generation paths into one function that receives an LLM.
- Apply the agreed API changes, including the `/v1` prefix (see Decisions made). These are breaking, so update `app/api_client.py` and `API.md` in the same checkpoint.
- Replace `_create_llm_for_model` in `api/custom_service.py` with a factory in `src/` that chooses the client by provider id, not by model-name prefix. Every provider's `base_url` lives in `config/providers.py`, including Groq's. An unknown provider is rejected instead of silently going to Groq.
- Prune the model catalogue. On 2026-11-13, 5 of the 9 Groq models listed (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen/qwen3.6-27b`, `groq/compound`, `groq/compound-mini`) returned 404 for the owner's key. Consider checking against each provider's `/models` endpoint.
- Replace the `thinking` prompt instruction with each provider's native reasoning control (Groq `reasoning_effort`). Reasoning models spend output tokens on reasoning, so the default `max_tokens` of 512 can cut answers short.
- The o-series OpenAI models accept only their default temperature, so do not send `temperature=0` to them.
- Return the figures the model actually saw. Add `figures` to the `/v1/query` response (and the `done` stream event): the image URLs sent in the prompt, in prompt order, each with its chunk and source document. Label each image in the prompt (`Figure F1, from <document>, chunk <n>`) so the model can cite `[F1]`. This is the baseline for citations; see Later phases for highlighted text.

**Checkpoint 3. Pipeline readability and known bugs.**
- Docstrings everywhere, remove all banner comments (about 75).
- Split `src/retrieval/generate.py` into prompt building, the LLM factory, and streaming.
- Fix:
  - only the first 5 chunks' text reaches the prompt while images come from all chunks;
  - streamed usage tokens reporting 0;
  - cost estimates hardcoded to Gemini prices.

**Checkpoint 4. Frontend separation.**
- `app/` gets its own small settings module.
- Add a `GET /providers` endpoint so the UI stops importing `config/providers.py`.
- Serve dataset browsing through the API instead of reading `json/` directly.
- Split `app/dashboard.py` into Streamlit pages plus shared components.
- Show the `figures` strip under each answer, with the figures the answer cites (`[F1]`) shown first.

**Checkpoint 5. Evaluation baseline.**
- Turn `evals/testing.json` into a small golden set covering text, table, figure, and "not stated" questions per document. The agent drafts each question with its expected answer and page; the owner reviews every one.
- One script reports retrieval hit rate and answer correctness.
- Record the baseline in `docs/TRADEOFFS.md` so later changes (chunking, embeddings, cache, the Postgres migration) are measured, not guessed.

**Checkpoint 6. Postgres + pgvector.** Host: Railway Postgres, tentatively; confirm before starting.
- Schema for documents, chunks (with `vector(768)`), images, queries, retrievals, answers, and feedback, with Alembic migrations.
- A one-off script copies ids, summaries, embeddings, and metadata out of Chroma into Postgres. No summarizer and no re-embedding.
- Swap the retriever to pgvector and confirm the evaluation numbers match Chroma's.
- Remove Chroma once they match.

**Checkpoint 7. Upload and ingest through the API.** This is the product goal of the phase.
- Keep each element's page number and bounding box from `unstructured` (`metadata.page_number`, `metadata.coordinates`) on the chunk. Highlighted citations depend on this, and it cannot be recovered later without re-extracting.
- `POST /documents` uploads a PDF and starts an ingestion job.
- `GET /documents/{id}` reports the job's status.
- Queries can be scoped to one or more documents.
- Research, then decide: the summarizer for new uploads (the Hugging Face credit is gone). Candidates are a hosted multimodal model taking 5 to 6 images per call, a local vision model, and multi-vector embeddings (`docs/TRADEOFFS.md`, section 3).
- Parser backend switch (`PARSER_BACKEND=local|hosted`), one adapter that turns local and hosted elements into the same shape, and figure cropping from page renders, because the hosted Parse API returns image boxes but no image bytes (`docs/TRADEOFFS.md`, section 1).

**Checkpoint 8. Semantic cache.**
- Cache answers keyed by question embedding similarity plus an exact match on part numbers. The key also includes the model, `top_k`, and the corpus version, and failures are never cached.
- Optional: a small router model that decides between a cache hit, retrieval, or a direct answer.

**Checkpoint 9. Deployment.**
- Bring Railway back with the watsonx and database variables.
- Set `API_BASE_URL` in Streamlit Cloud secrets.
- Document the deploy in the README.

## Later phases (not now)

Recorded so the current work does not block them. Do not start these until the checkpoints above are done.

- **React frontend.** A TypeScript and React web app (Next.js or Vite, decided then) replaces Streamlit as the product UI, and Streamlit stays as an internal inspection tool. It is possible only because the UI talks to the backend over HTTP alone, which Checkpoint 4 finishes. The owner will also use it to practise JavaScript fundamentals.
- **Highlighted citations.** The UI renders the source PDF page (for example with pdf.js) and highlights the exact passage or table cell an answer came from, using the page numbers and bounding boxes kept in Checkpoint 7. This goes further than citing a file name.
- **PCB design agent.** The long-term direction described at the top of this file.

## Things not to do

- Do not start the PCB design agent, MCP servers, web search, or LangGraph work in this phase.
- Do not change public API shapes, delete files, add dependencies, or change models without asking.
- Do not commit `.env`, `.streamlit/secrets.toml`, or anything containing a key.
- Do not re-index or re-summarize the corpus without saying so first. It costs API calls and rewrites stored data.
- Do not leave dead code "for later". Delete it; git history keeps it.
