"""Text and example payloads shown in the interactive API docs (``/docs``).

Shared by the retrieval and query routers. Nothing here affects how a request
is handled; it only shapes the OpenAPI schema FastAPI generates.
"""

from __future__ import annotations

REQUEST_BODY_EXAMPLES = [
    {
        "summary": "LM2596 pin configuration",
        "value": {"query": "Tell me about the pin configuration of the LM2596.", "top_k": 10},
    },
    {
        "summary": "LM317 pin configuration",
        "value": {"query": "Tell me about the pin configuration of the LM317.", "top_k": 5},
    },
]

QUERY_PARAM_EXAMPLES = [
    "Tell me about the pin configuration of the LM2596.",
    "Tell me about the pin configuration of the LM317.",
]

RICH_DESCRIPTION_COMMON = """
Ask a question about the indexed PDFs. The service:

1. embeds the question with the same model used at ingest time,
2. retrieves the `top_k` most relevant chunks from Chroma,
3. (for `/query` and `/query/stream`) asks the multimodal LLM to answer
   using the chunk text, tables and figures.

### Try it

```json
{
  "query": "Tell me about the pin configuration of the LM317.",
  "top_k": 10
}
```

### Response

Every response includes a `response_id`; chunk figures can be loaded over
HTTP via `GET /images/{response_id}/{index}`.
"""


def request_body_examples(schema_name: str, examples: list[dict]) -> dict:
    """Build an ``openapi_extra`` requestBody with a named Examples dropdown.

    Args:
        schema_name: The Pydantic model name the body refers to.
        examples: Items of the form ``{"summary": str, "value": dict}``.

    Returns:
        A dict to pass as a route's ``openapi_extra``.
    """
    return {
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {"$ref": f"#/components/schemas/{schema_name}"},
                    "examples": {
                        f"example_{i + 1}": {
                            "summary": ex.get("summary", f"Example {i + 1}"),
                            "value": ex.get("value", ex),
                        }
                        for i, ex in enumerate(examples)
                    },
                }
            }
        }
    }
