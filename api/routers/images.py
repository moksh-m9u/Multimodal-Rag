"""Figure endpoint: serves one image referenced by a chunk's ``image_urls``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

router = APIRouter(tags=["images"])


@router.get(
    "/images/{response_id}/{index}",
    summary="Fetch a figure as JPEG",
    description=(
        "Serves one figure referenced by a chunk's ``image_urls``.\n\n"
        "- ``response_id`` comes from the retrieve/query response and groups "
        "all figures of that response.\n"
        "- ``index`` is the **global, 0-based** position of the image across "
        "the whole response (chunk 2 of a 2-image chunk 1 starts at index 2).\n\n"
        "```\n"
        "GET /images/79c107621a50474989f2c3eab59beb7a/1\n"
        "```\n"
        "Returns the figure as ``image/jpeg``. Entries are cached for 10 "
        "minutes and expire after ~30; expired or unknown ids return 404."
    ),
    response_description="image/jpeg bytes of the requested figure",
    responses={404: {"description": "Unknown response_id or expired / out-of-range index"}},
)
async def serve_image(response_id: str, index: int, request: Request) -> Response:
    """Serve one figure from a previous retrieve/query response.

    Entries expire after ~30 minutes; expired or unknown ids return 404.
    """
    raw = request.app.state.service.images.get(response_id, index)
    if raw is None:
        raise HTTPException(status_code=404, detail="Image not found or expired")
    return Response(
        content=raw,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=600"},
    )
