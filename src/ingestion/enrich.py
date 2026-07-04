"""Content-type separation and AI-enhanced summary generation."""

import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from config.settings import (
    HF_TOKEN,
    ENHANCEMENT_MODEL,
    ENHANCEMENT_BASE_URL,
    ENHANCEMENT_TEMPERATURE,
    ENHANCEMENT_MAX_TOKENS,
)


def separate_content_types(chunk: Any) -> dict:
    """Analyze what types of content are in a chunk.

    Returns a dict with keys: text, tables, images, types.
    """
    content_data: dict = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"],
    }

    orig_elements = getattr(chunk.metadata, "orig_elements", None)
    if orig_elements is None:
        return content_data

    for element in orig_elements:
        element_type = type(element).__name__

        if element_type == "Table":
            content_data["types"].append("table")
            table_html = getattr(
                element.metadata, "text_as_html", element.text
            )
            content_data["tables"].append(table_html)

        elif element_type == "Image":
            img_b64 = getattr(element.metadata, "image_base64", None)
            if img_b64:
                content_data["types"].append("image")
                content_data["images"].append(img_b64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


def _create_enhancement_llm() -> ChatOpenAI:
    """Create the LLM used for content enhancement."""
    return ChatOpenAI(
        model=ENHANCEMENT_MODEL,
        base_url=ENHANCEMENT_BASE_URL,
        api_key=HF_TOKEN,
        temperature=ENHANCEMENT_TEMPERATURE,
        max_tokens=ENHANCEMENT_MAX_TOKENS,
    )


def _build_enhancement_prompt(text: str, tables: list[str]) -> str:
    """Build the text prompt for AI-based content enhancement."""
    prompt = (
        "You are creating a searchable description for document content retrieval.\n\n"
        "CONTENT TO ANALYZE:\n"
        "TEXT CONTENT:\n"
        f"{text}\n\n"
    )

    if tables:
        prompt += "TABLES:\n"
        for i, table in enumerate(tables):
            prompt += f"Table {i + 1}:\n{table}\n\n"

    prompt += (
        "YOUR TASK:\n"
        "Generate a comprehensive, searchable description that covers:\n\n"
        "1. Key facts, numbers, and data points from text and tables\n"
        "2. Main topics and concepts discussed\n"
        "3. Questions this content could answer\n"
        "4. Visual content analysis (charts, diagrams, patterns in images)\n"
        "5. Alternative search terms users might use\n\n"
        "Make it detailed and searchable - prioritize findability over brevity.\n\n"
        "SEARCHABLE DESCRIPTION:"
    )

    return prompt


def create_ai_enhanced_summary(
    text: str, tables: list[str], images: list[str]
) -> str:
    """Create an AI-enhanced searchable summary for mixed content.

    Args:
        text: Raw OCR text.
        tables: List of HTML table strings.
        images: List of base64-encoded image strings.

    Returns:
        The enhanced summary text, or a fallback summary on failure.
    """
    try:
        llm = _create_enhancement_llm()
        prompt_text = _build_enhancement_prompt(text, tables)

        message_content: list[dict] = [{"type": "text", "text": prompt_text}]
        for b64 in images:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        print(f"    AI summary failed: {e}")
        fallback = f"{text[:300]}..."
        if tables:
            fallback += f" [Contains {len(tables)} table(s)]"
        if images:
            fallback += f" [Contains {len(images)} image(s)]"
        return fallback


def summarise_chunks(chunks: list) -> list[Document]:
    """Process all chunks with AI summaries and wrap them as LangChain Documents.

    Each Document stores the enhanced content as page_content and the full
    original data (raw text, tables, images) as JSON in metadata.

    Args:
        chunks: List of unstructured chunks.

    Returns:
        List of langchain Documents ready for vector storage.
    """
    print("Processing chunks with AI summaries...")
    langchain_documents: list[Document] = []

    for i, chunk in enumerate(chunks):
        current = i + 1
        total = len(chunks)
        print(f"  Processing chunk {current}/{total}")

        content_data = separate_content_types(chunk)
        print(
            f"    Types: {content_data['types']}  |  "
            f"Tables: {len(content_data['tables'])}  |  "
            f"Images: {len(content_data['images'])}"
        )

        if content_data["tables"] or content_data["images"]:
            print("    -> Creating AI summary for mixed content...")
            enhanced = create_ai_enhanced_summary(
                content_data["text"],
                content_data["tables"],
                content_data["images"],
            )
            print(f"    -> Enhanced: {enhanced[:150]}...")
        else:
            print("    -> Using raw text (no tables/images)")
            enhanced = content_data["text"]

        doc = Document(
            page_content=enhanced,
            metadata={
                "original_content": json.dumps(
                    {
                        "raw_text": content_data["text"],
                        "tables_html": content_data["tables"],
                        "images_base64": content_data["images"],
                    }
                )
            },
        )
        langchain_documents.append(doc)

    print(f"Processed {len(langchain_documents)} chunks")
    return langchain_documents
