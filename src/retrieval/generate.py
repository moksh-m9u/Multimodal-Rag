"""Answer generation using a multimodal LLM."""

import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from config.settings import GENERATION_MODEL, GENERATION_TEMPERATURE


def _extract_original_data(chunk: Document) -> dict:
    """Extract and parse original_content metadata from a chunk."""
    raw = chunk.metadata.get("original_content")
    if raw is None:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _build_text_prompt(chunks: list[Document], query: str) -> str:
    """Build the text portion of the prompt from retrieved chunks."""
    parts = [
        f"Based on the following documents, please answer this question: {query}",
        "",
        "CONTENT TO ANALYZE:",
        "",
    ]

    for i, chunk in enumerate(chunks):
        parts.append(f"--- Document {i + 1} ---")

        original_data = _extract_original_data(chunk)
        raw_text = original_data.get("raw_text", "")
        if raw_text:
            parts.append(f"TEXT:\n{raw_text}\n")

        tables_html = original_data.get("tables_html", [])
        if tables_html:
            parts.append("TABLES:")
            for j, table in enumerate(tables_html):
                parts.append(f"Table {j + 1}:\n{table}\n")

        parts.append("")

    parts.append(
        "Please provide a clear, comprehensive answer using the text, tables, and "
        "images above. If the documents don't contain sufficient information to answer "
        'the question, say "I don\'t have enough information to answer that question '
        'based on the provided documents."'
    )
    parts.append("")
    parts.append("ANSWER:")

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

        if verbose:
            image_count = sum(
                1 for x in message_content if x["type"] == "image_url"
            )
            print(f"Message parts: {len(message_content)}")
            print(f"Images attached: {image_count}")

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        error_msg = f"Answer generation failed: {e}"
        if verbose:
            print(error_msg)
        return f"Sorry, could not complete response due to: {e}"
