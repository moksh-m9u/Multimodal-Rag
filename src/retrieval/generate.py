"""Answer generation using a multimodal LLM."""

import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from config.settings import GENERATION_MODEL, GENERATION_TEMPERATURE


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
        llm = _create_llm()
        message_content = _build_message_content(chunks, query)
        _save_prompt_debug(message_content)

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
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


def generate_answer_stream(
    chunks: list[Document], query: str, verbose: bool = False
):
    """Generate a streaming answer using the multimodal LLM.

    Yields answer tokens as they are generated by the LLM.
    """
    try:
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

        message = HumanMessage(content=message_content)
        for chunk in llm.stream([message]):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        error_msg = f"Answer generation failed: {e}"
        print(error_msg)
        yield f"Sorry, could not complete response due to: {e}"
