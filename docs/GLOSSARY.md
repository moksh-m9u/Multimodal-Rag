# Glossary

Plain-language explanations of the technical terms used in this repository. Each entry says what the term means and where it shows up here. Add a term in the same change that first uses it.

## Retrieval and embeddings

**Embedding.** A list of numbers (here 768) that represents the meaning of a piece of text. Texts with similar meaning get vectors that point in similar directions. Here: `src/embed.py`, watsonx Granite.

**Cosine similarity.** How close two embeddings are, measured by the angle between them: 1 means the same direction, 0 means unrelated. Chroma and pgvector both rank chunks by it here.

**Vector store.** A database that stores embeddings and finds the nearest ones to a query embedding. Here: Chroma today, Postgres with pgvector after Checkpoint 6.

**HNSW (Hierarchical Navigable Small World).** The index that makes nearest-neighbour search fast. It builds a layered graph of vectors and walks it towards the query instead of comparing against every vector. The result is approximate but very close to exact. Chroma and pgvector both use it.

**MMR (Maximal Marginal Relevance).** A way to pick results that are relevant to the question but different from each other. Here it fetches the 20 nearest chunks and keeps 10, so the prompt does not get five copies of the same table. `src/retrieval/search.py`.

**Hybrid search.** Running a keyword search and a vector search together and merging the results. Keyword search catches exact tokens such as `TLV1117-33` or `VREF` that embeddings blur. Planned once the data is on Postgres.

**RRF (Reciprocal Rank Fusion).** A simple way to merge two ranked lists: each result scores `1 / (k + rank)` in every list it appears in, and the scores are added. It needs no tuning of score scales, which is why hybrid search usually uses it.

**Multi-vector embeddings.** Storing more than one vector per chunk (for example one for the raw text and one for the summary) and matching a question against any of them. Better recall, more storage.

## Ingestion

**Partitioning.** Splitting a PDF into typed elements (`Title`, `NarrativeText`, `Table`, `Image`, and so on). Here: `unstructured`, `src/ingestion/extract.py`.

**hi_res strategy.** The `unstructured` mode that renders each page as an image and runs a layout detection model on it to find tables and figures. Accurate but slow and heavy, which is why production uses the hosted API.

**Chunking by title.** Grouping elements into chunks that start at each section heading, so a chunk follows the document's own sections. `src/ingestion/chunk.py`.

**Coordinates (bounding box).** The rectangle an element occupies on its page. `unstructured` reports it as four corner points plus the coordinate system (for example `PixelSpace` with the page rendered at 2125 x 2750). Needed for cropping figures and for highlighted citations.

**Enrichment (summary).** A vision model's written description of a chunk, including its tables and figures. The summary is what gets embedded; the original is what the answer is built from.

## API and serving

**Lifespan.** The FastAPI hook that runs code once when the server starts and once when it stops. Here it loads the embedding client and the vector store a single time instead of per request. `api/main.py`.

**SSE (Server-Sent Events).** A way for a server to send a stream of small messages over one HTTP response. Used to stream the answer token by token from `/query/stream`.

**Event loop.** The single thread that runs all `async` endpoints in FastAPI. Blocking work on it stalls every request, which is why synchronous LangChain calls run in `asyncio.to_thread`.

**Multipart form data.** The HTTP body format for uploading files together with form fields. The hosted Parse API and the planned `POST /v1/documents` both use it.

**202 Accepted.** An HTTP status meaning "the request was accepted and will finish later". Used for long jobs such as ingestion, with a status URL to check.
