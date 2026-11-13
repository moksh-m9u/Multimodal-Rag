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

## Docker and deployment

**Image.** A read-only snapshot of a filesystem plus a start command. `docker build` produces it from the `Dockerfile`; `docker run` starts a container from it. Here: `multimodal-rag-api`.

**Layer.** One step of the `Dockerfile` (`RUN`, `COPY`) stored as a diff on top of the previous step. Docker reuses a layer when its inputs did not change, which is why `requirements-api.txt` is copied and installed before the code: a code change does not reinstall the packages.

**Build context.** The folder `docker build` sends to the Docker engine (the `.` at the end of the command). `COPY . .` can only copy files that are in the context.

**.dockerignore.** A file listing paths to leave out of the build context. Here it is an allowlist: `*` ignores everything and the `!` lines re-include `api/`, `src/`, `config/`, `dbv2/`, and `requirements-api.txt`.

**Allowlist and denylist.** An allowlist names what is permitted and rejects the rest; a denylist names what is rejected and permits the rest. An allowlist is safer by default, because something new is left out until someone adds it.

**Container.** A running process started from an image, isolated from the host with its own filesystem, network, and process list. It is not a virtual machine: it shares the host's kernel. Anything written inside it is lost when it is removed, unless it is written to a volume.

**Dockerfile.** The recipe `docker build` follows to produce an image: one instruction per line, run top to bottom. Here: `Dockerfile` at the repository root.

**Base image (`FROM`).** The image a Dockerfile starts from. Here `python:3.13-slim`: Debian with Python 3.13 and few system packages. `slim` is smaller than the full image and, unlike `alpine`, can install ordinary prebuilt Python wheels.

**Tag.** The label after the colon in an image name (`multimodal-rag-api:latest`, `python:3.13-slim`). A tag can be moved to a newer image; a digest (`@sha256:...`) names one exact image forever.

**Registry.** A server that stores images so other machines can pull them. Docker Hub is the default one. `docker push` uploads, `docker pull` downloads, and only layers the other side lacks are transferred.

**Layer cache.** Docker reuses a layer from the previous build when the instruction and its input files are unchanged. The first changed layer and every layer after it are rebuilt, so instructions go from least often changed (system packages, dependencies) to most often changed (application code).

**`RUN`, `COPY`, `ENV`, `WORKDIR`.** Build-time instructions. `RUN` executes a command and saves the result as a layer. `COPY` brings files from the build context into the image. `ENV` sets an environment variable for later instructions and for the running container. `WORKDIR` sets the folder later instructions and the start command run in (`/app` here).

**`CMD` and `ENTRYPOINT`.** What the container runs when it starts. `CMD` is the default and is replaced by any arguments given to `docker run`; `ENTRYPOINT` is fixed and receives those arguments. Here `CMD` starts `uvicorn`.

**Exec form and shell form.** `CMD ["uvicorn", "api.main:app"]` (exec form) starts the program directly. `CMD uvicorn api.main:app` (shell form) starts `sh -c` with the program under it. Only a shell expands `${PORT}`, which is why the `CMD` here calls `sh -c` explicitly.

**PID 1.** The first process in a container. `docker stop` sends it `SIGTERM` and kills the container after 10 seconds if it has not exited. If PID 1 is a shell that does not pass the signal on, the server never gets the chance to shut down cleanly.

**`EXPOSE`.** A note in the image saying which port the server listens on. It opens nothing by itself; `docker run -p 8000:8000` (host port, then container port) does the publishing.

**`HEALTHCHECK`.** A command Docker runs inside the container on a timer; exit code 0 marks the container `healthy`. Here it requests `/health`. Hosting platforms often ignore it and use their own health check setting.

**Environment file (`--env-file`).** A file of `NAME=value` lines that `docker run` turns into environment variables in the container. Docker takes each line literally, so quotes become part of the value. It keeps secrets out of the image, which anyone who can pull the image could otherwise read.

**Volume.** A folder that lives outside the container's own filesystem and survives when the container is removed. Not used here yet; the Chroma store is baked into the image instead.

**Multi-stage build.** A Dockerfile with several `FROM` lines. An early stage holds compilers and build tools, and the final stage copies only the finished result out of it, so the tools never reach the shipped image.
