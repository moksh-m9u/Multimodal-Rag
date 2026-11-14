"""Text and example payloads shown in the interactive API docs (``/docs``).

Shared by the retrieval and query routers. Nothing here affects how a request
is handled; it only shapes the OpenAPI schema FastAPI generates.
"""

from __future__ import annotations

# Named examples for request bodies, in the shape FastAPI's
# ``Body(openapi_examples=...)`` expects: each key is one entry in the
# Examples dropdown on /docs, ``summary`` is its label, ``value`` the payload.
QUERY_BODY_EXAMPLES = {
    "lm2596_pins": {
        "summary": "LM2596 pin configuration",
        "value": {"query": "Tell me about the pin configuration of the LM2596.", "top_k": 10},
    },
    "lm317_pins": {
        "summary": "LM317 pin configuration",
        "value": {"query": "Tell me about the pin configuration of the LM317.", "top_k": 5},
    },
}

CUSTOM_QUERY_BODY_EXAMPLES = {
    "gemini": {
        "summary": "Custom Gemini query",
        "value": {
            "query": "Tell me about the pin configuration of the LM2596.",
            "top_k": 10,
            "provider": "gemini",
            "model": "gemini-3.5-flash-lite",
            "api_key": "YOUR_GEMINI_API_KEY",
            "temperature": 0.0,
            "max_tokens": 512,
        },
    },
}

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

