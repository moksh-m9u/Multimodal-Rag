"""Answer generation using a multimodal LLM."""

import base64 as _b64
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langsmith import traceable
from langsmith.run_helpers import set_tracing_parent

from config.settings import GENERATION_MODEL, GENERATION_TEMPERATURE
from src.retrieval.search import retrieve_chunks
from src.tracing import summarize_chunks


def _extract_original_data(chunk: Document) -> dict:
    """Extract and parse original_content metadata from a chunk.

    Handles both nested and flattened ChromaDB metadata schemas.
    """
    raw = chunk.metadata.get("original_content")
    if raw is not None:
        if isinstance(raw, str):
            return json.loads(raw)
        return raw

    raw_text = chunk.metadata.get("raw_text", "")
    tables_raw = chunk.metadata.get("tables_html", "[]")
    tables_html = json.loads(tables_raw) if isinstance(tables_raw, str) else tables_raw

    images_base64: list[str] = []
    image_paths_raw = chunk.metadata.get("image_paths", "[]")
    image_paths = json.loads(image_paths_raw) if isinstance(image_paths_raw, str) else image_paths_raw

    from pathlib import Path as _P
    import base64 as _b64

    project_root = _P(__file__).resolve().parent.parent.parent
    for p in image_paths:
        clean = p.lstrip("./")
        img_path = project_root / clean
        try:
            img_bytes = img_path.read_bytes()
            images_base64.append(_b64.b64encode(img_bytes).decode())
        except Exception as e:
            print(f"  [WARN] Could not read image for generation: {img_path} ({e})")

    return {
        "raw_text": raw_text,
        "tables_html": tables_html,
        "images_base64": images_base64,
    }


def _build_text_prompt(chunks: list[Document], query: str) -> str:
    """Build the text portion of the prompt from retrieved chunks.

    Sends enhanced summaries as primary context, with HTML tables appended.
    Images are sent separately via _collect_images.
    """
    parts = [
        "You are given retrieved chunks from a technical document.\n"
        "Each chunk contains:\n"
        "- a searchable summary of the original content,\n"
        "- optional HTML tables,\n"
        "- optional attached figures/images.\n\n"
        "Treat the summaries as authoritative representations of the document.\n"
        "Do NOT claim information is absent unless you have examined ALL provided chunks.\n"
        "If a chunk explicitly contains the answer, answer directly using that chunk.\n"
        "Use both the text summaries and the attached images when answering.\n",
        f"QUESTION: {query}\n",
        "RETRIEVED CONTEXT:",
        "",
    ]

    # DIAGNOSTIC: only top 1 chunk, no images — test if Gemini receives text
    for i, chunk in enumerate(chunks[:5]):
        enhanced = chunk.page_content
        if enhanced:
            parts.append(f"--- Chunk {i + 1} ---")
            parts.append(f"SUMMARY:\n{enhanced.strip()}\n")

        original_data = _extract_original_data(chunk)

        tables_html = original_data.get("tables_html", [])
        if tables_html:
            parts.append("TABLES:")
            for j, table in enumerate(tables_html):
                parts.append(f"Table {j + 1}:\n{table}\n")

        parts.append("")

    parts.append(
        "The images above are figures from the document. "
        "Use the summaries, tables, and images together to provide a detailed answer."
    )

    return "\n".join(parts)


def _collect_images(chunks: list[Document]) -> list[dict]:
    """Collect all base64 images from chunks into message content blocks."""
    image_blocks: list[dict] = []
    for chunk in chunks:
        original_data = _extract_original_data(chunk)
        for b64 in original_data.get("images_base64", []):
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
    return image_blocks


def _build_message_content(chunks: list[Document], query: str) -> list[dict]:
    """Build the full multimodal message content list."""
    text_prompt = _build_text_prompt(chunks, query)
    content: list[dict] = [{"type": "text", "text": text_prompt}]
    content.extend(_collect_images(chunks))
    return content


def _create_llm():
    """Create the Gemini LLM instance."""
    print(f"\n  Using model: {GENERATION_MODEL}")
    return ChatGoogleGenerativeAI(
        model=GENERATION_MODEL,
        temperature=GENERATION_TEMPERATURE,
    )


@traceable(run_type="llm", name="GenerateAnswer")
def generate_answer(
    chunks: list[Document], query: str, verbose: bool = False
) -> str:
    """Generate a final answer using the multimodal LLM.

    Args:
        chunks: Retrieved document chunks.
        query: The original user query.
        verbose: If True, prints debug info.

    Returns:
        The generated answer string.
    """
    try:
        chunk_summaries = summarize_chunks(chunks)
        total_images = sum(c.get("image_count", 0) for c in chunk_summaries)
        total_tables = sum(c.get("table_count", 0) for c in chunk_summaries)

        llm = _create_llm()
        message_content = _build_message_content(chunks, query)
        _save_prompt_debug(message_content)

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        print(
            f"\n  LangSmith trace: {len(chunks)} chunks, "
            f"{total_images} images, {total_tables} tables"
        )
        return response.content

    except Exception as e:
        error_msg = f"Answer generation failed: {e}"
        if verbose:
            print(error_msg)
        return f"Sorry, could not complete response due to: {e}"


def _save_prompt_debug(message_content: list[dict]) -> None:
    """Dump the full prompt to last_prompt.txt for inspection."""
    from pathlib import Path as _P
    text_content = next(x["text"] for x in message_content if x["type"] == "text")
    image_count = sum(1 for x in message_content if x["type"] == "image_url")
    _P("last_prompt.txt").write_text(text_content, encoding="utf-8")
    print(f"\n  Prompt saved to last_prompt.txt ({len(text_content)} chars, {image_count} images)\n")


def _collect_image_attachments(chunks: list[Document]) -> dict:
    """Gather images from chunks for LangSmith trace attachments.

    Handles both file-path and base64 storage schemas via
    ``_extract_original_data`` (which resolves file paths to base64).
    """
    attachments = {}
    img_idx = 0
    for chunk in chunks:
        od = _extract_original_data(chunk)
        for b64_str in od.get("images_base64", []):
            try:
                attachments[f"chunk_img_{img_idx}"] = (
                    "image/jpeg",
                    _b64.b64decode(b64_str),
                )
                img_idx += 1
            except Exception:
                pass

    return attachments


@traceable(run_type="chain", name="AnswerQuery", dangerously_allow_filesystem=True)
def answer_query(retriever, query: str, run_tree=None) -> str:
    """Retrieve chunks then generate an answer under a single LangSmith trace.

    Images from retrieved chunks are attached to the trace so they render
    in the LangSmith UI.  ``run_tree`` is injected by LangSmith.
    """
    chunks = retrieve_chunks(retriever, query)

    if run_tree is not None:
        attachments = _collect_image_attachments(chunks)
        if attachments:
            run_tree.attachments = attachments

    return generate_answer(chunks, query)


@traceable(run_type="chain", name="AnswerQuery", dangerously_allow_filesystem=True)
def answer_query_stream(retriever, query: str, run_tree=None):
    """Retrieve chunks then stream an answer under a single LangSmith trace.

    Yields answer tokens.  Chunks are available via the returned generator's
    ``chunks`` attribute after retrieval completes.  ``run_tree`` is injected
    by LangSmith.
    """
    chunks = retrieve_chunks(retriever, query)

    if run_tree is not None:
        attachments = _collect_image_attachments(chunks)
        if attachments:
            run_tree.attachments = attachments

    class _StreamWrapper:
        def __init__(self, gen, chunks, parent=None):
            self._gen = gen
            self.chunks = chunks
            self._parent = parent
        def __iter__(self):
            return self
        def __next__(self):
            if self._parent is not None:
                with set_tracing_parent(self._parent):
                    return next(self._gen)
            return next(self._gen)

    return _StreamWrapper(
        generate_answer_stream(chunks, query),
        chunks,
        run_tree,
    )


@traceable(run_type="llm", name="GenerateAnswerStream")
def generate_answer_stream(
    chunks: list[Document], query: str, verbose: bool = False
):
    """Generate a streaming answer using the multimodal LLM.

    Yields answer tokens as they are generated by the LLM.
    """
    try:
        chunk_summaries = summarize_chunks(chunks)
        total_images = sum(c.get("image_count", 0) for c in chunk_summaries)
        total_tables = sum(c.get("table_count", 0) for c in chunk_summaries)

        llm = _create_llm()

        print(f"\n  {len(chunks)} chunks received by generation")
        for ci, c in enumerate(chunks[:5]):
            pc = c.page_content or ""
            od = _extract_original_data(c)
            print(f"    Chunk {ci+1}: summary={len(pc)} chars, "
                  f"tables={len(od.get('tables_html',[]))}, "
                  f"images={len(od.get('images_base64',[]))}")

        message_content = _build_message_content(chunks, query)
        _save_prompt_debug(message_content)

        print(
            f"\n  LangSmith trace: {len(chunks)} chunks, "
            f"{total_images} images, {total_tables} tables"
        )

        message = HumanMessage(content=message_content)
        for chunk in llm.stream([message]):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        error_msg = f"Answer generation failed: {e}"
        print(error_msg)
        yield f"Sorry, could not complete response due to: {e}"
