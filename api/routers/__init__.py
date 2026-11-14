"""HTTP endpoints, one module per purpose.

Each module defines an ``APIRouter`` that ``api/main.py`` includes:
``system`` (index and health), ``retrieval`` (chunks only), ``query``
(answers, streamed or not) and ``images`` (figure bytes).
"""
