"""Smoke test: trace a single image to LangSmith via wrap_openai.

Run:
    python -m scripts.test_single_image_trace

Then check the printed LangSmith URL in your browser.
"""

import base64
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from langsmith.wrappers import wrap_openai

from config.settings import HF_TOKEN, ENHANCEMENT_MODEL, ENHANCEMENT_BASE_URL


IMAGE_PATH = "dbv2/images/AN699chunks_c1_img0.jpg"
MIME_TYPE = "image/jpeg"


def main() -> None:
    with open(IMAGE_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    client = wrap_openai(
        OpenAI(api_key=HF_TOKEN, base_url=ENHANCEMENT_BASE_URL),
        chat_name="SingleImageTrace",
    )

    response = client.chat.completions.create(
        model=ENHANCEMENT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in a few words."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{MIME_TYPE};base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    print("\nAnswer:", response.choices[0].message.content)

    from langsmith import Client as LSClient

    ls = LSClient()
    runs = ls.list_runs(project_name="Multimodal Rag", run_type="llm", limit=1)
    for run in runs:
        print("\nLangSmith run id:", run.id)
        print(
            "Trace URL: "
            f"https://smith.langchain.com/o/1/projects/p/{run.id}?display=long"
        )


if __name__ == "__main__":
    main()
