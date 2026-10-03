"""Multimodal RAG Chunk Inspector -- Debug and Validate Chunks."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import base64
import io
import re
import time
from pathlib import Path
from typing import Any, Optional

import streamlit as st


def _merge_streamlit_secrets() -> None:
    """Expose Streamlit secrets as environment variables.

    Locally the app reads ``.env`` (gitignored).  On Streamlit Cloud the
    equivalent is ``st.secrets``, but not every runtime surfaces those as
    process environment variables for third-party SDKs (LangSmith, Hugging
    Face, Gemini).  Copying them into ``os.environ`` before the settings and
    tracing modules are imported guarantees all backends read the same config
    in both environments.
    """
    try:
        for key, value in st.secrets.items():
            os.environ.setdefault(str(key), str(value))
    except Exception:
        pass

_merge_streamlit_secrets()

import pandas as pd
from PIL import Image
import plotly.express as px

from config.settings import API_BASE_URL
from config.providers import ALL_PROVIDERS, list_models_for_provider
from src.logger import get_logger

from api_client import health as api_health
from api_client import stream as api_stream
from api_client import query_custom

logger = get_logger(__name__)

PAGE_TITLE = "Multimodal RAG Chunk Inspector"
LAYOUT = "wide"

# ---------------------------------------------------------------------------
# Landing page content
# ---------------------------------------------------------------------------
# Shown instead of the "upload a JSON" placeholder whenever no datasheet has
# been selected, so a first-time visitor lands on an explanation of the project
# rather than on an empty screen.

GITHUB_URL = "https://github.com/moksh-m9u/Multimodal-Rag"
LIVE_APP_URL = "https://multimodal-rag-complex.streamlit.app/"
EVAL_DASHBOARD_URL = (
    "https://multimodal-rag-iquyr6mkijuujmcr8cmikc.streamlit.app/"
)

PROJECT_LINKS: list[tuple[str, str]] = [
    ("GitHub", GITHUB_URL),
    ("Live App", LIVE_APP_URL),
    ("Evaluation Dashboard", EVAL_DASHBOARD_URL),
    ("API Reference", f"{GITHUB_URL}/blob/main/API.md"),
    ("LangSmith", "https://smith.langchain.com/"),
]

CORE_STACK: list[str] = [
    "Python",
    "FastAPI",
    "Streamlit",
    "LangChain",
    "LangSmith",
    "LLMOps",
    "ChromaDB",
    "Tesseract OCR",
    "GLM-4.5V",
    "Gemini",
    "Hugging Face",
]

SUPPORT_STACK: list[str] = [
    "unstructured",
    "Granite Embeddings",
    "MMR Retrieval",
    "Pandas",
    "Plotly",
    "SSE Streaming",
    "Docker",
    "pytest",
]

# (step, title, body, tags)
PIPELINE_STEPS: list[tuple[str, str, str, list[str]]] = [
    (
        "Step 01",
        "Multimodal ingestion of electronic datasheets",
        "Datasheets hide their answers inside pictures: pinout diagrams, application "
        "circuits, characteristic curves, timing waveforms, package drawings. "
        "<code>partition_pdf(strategy=\"hi_res\")</code> splits every datasheet into typed "
        "elements — narrative text, structured HTML tables and base64 image blocks — while "
        "preserving page numbers and reading order so each figure stays associated with the "
        "section that explains it.",
        ["PDF in", "unstructured", "hi_res strategy", "OCR"],
    ),
    (
        "Step 02",
        "OCR + vision-LLM enhanced representations",
        "Tesseract OCR recovers text that physically lives inside images — pin labels, axis "
        "values, curve annotations. Then <b>GLM-4.5V</b>, a vision language model, writes a "
        "multimodal summary of every image- and table-rich chunk. That summary becomes "
        "<code>enhanced_content</code>: a text-only retrieval handle for knowledge that was "
        "originally visual, so a figure can now be found by asking about it.",
        ["Tesseract OCR", "GLM-4.5V", "enhanced_content", "+1024 tok"],
    ),
    (
        "Step 03",
        "Vector storage and diversity-aware retrieval",
        "Each chunk — enhanced content, raw text, tables and images — is embedded with "
        "Granite embeddings and persisted to <b>ChromaDB</b> using cosine similarity over HNSW. "
        "Retrieval uses <b>MMR</b> (top-10 answers from a 20-chunk candidate pool) so results "
        "are relevant <i>and</i> diverse instead of ten near-duplicate paragraphs. The same "
        "chunks are exported to JSON, which is what powers the inspector pages.",
        ["Granite embeddings", "ChromaDB", "cosine HNSW", "MMR k=10 / fetch_k=20"],
    ),
    (
        "Step 04",
        "Chunk Inspector &amp; Dataset Analytics",
        "A granular inspection dashboard where every chunk is opened independently: raw OCR "
        "text, AI-generated summary, decoded images, parsed tables, full metadata and "
        "cross-modal associations. Each chunk gets a 0–100 <b>health score</b> from content "
        "completeness, so weak chunks can be found before they hurt a query. Dataset Analytics "
        "adds distributions and top-20 lists over the whole corpus.",
        ["0–100 health score", "per-chunk drill-down", "histograms", "top-20 lists"],
    ),
    (
        "Step 05",
        "Grounded multimodal query &amp; retrieve",
        "The <b>FastAPI</b> backend runs embed → search → context assembly → generation. The "
        "top-10 relevant chunks are passed to the generation model together with their actual "
        "images and tables, not as flattened text, so answers can describe a pinout or read a "
        "curve. Answers stream back over SSE, with token usage, latency and estimated cost. "
        "Any provider and model can be selected in the sidebar with <b>Bring Your Own API Key</b>.",
        ["Top-10 chunks", "BYO API key", "Gemini / Groq / OpenAI / HF", "thinking mode"],
    ),
    (
        "Step 06",
        "LangSmith observability &amp; LLMOps",
        "End-to-end tracing covers embedding, retrieval, multimodal context assembly and "
        "generation. Every run is inspectable in LangSmith: the chunks that were retrieved, the "
        "model calls made, inputs and outputs, latency, token usage and the full execution flow — "
        "so a bad answer can be traced back to the exact stage that caused it.",
        ["end-to-end traces", "latency", "token usage", "inputs / outputs"],
    ),
    (
        "Step 07",
        "LLM-as-a-Judge evaluation framework",
        "A judge model scores both halves of the system: retrieval (relevance, coverage, "
        "ranking, redundancy, noise) and generation (correctness, faithfulness, completeness, "
        "conciseness, image utilisation). Across <b>13 curated datasheet questions</b> the "
        "system reaches an <b>89.8/100</b> average overall score with a <b>92% pass rate</b>, "
        "backed by a live evaluation dashboard with score distribution and failure analysis.",
        ["89.8 / 100 avg", "92% pass rate", "13 cases", "failure analysis"],
    ),
]

EVAL_METRICS: list[tuple[str, str, str]] = [
    ("89.8", "Avg overall judge score", "of 100 across 13 evaluation cases"),
    ("92%", "Pass rate", "12 of 13 cases graded Pass"),
    ("87.0", "Avg retrieval score", "relevance, coverage, ranking, redundancy, noise"),
    ("93.3", "Avg generation score", "correctness, faithfulness, completeness, conciseness"),
]

DATASET_LABELS: dict[str, str] = {
    "AN699chunks.json": "AN699",
    "LM555TImer.json": "LM555 Timer",
    "lm2596chunks.json": "LM2596",
    "lm317chunks.json": "LM317",
    "tlv1117.pdfchunks_export.json": "TLV1117",
}


def _dataset_display_name(filename: str) -> str:
    """Human-friendly label for a chunk export file."""
    if filename in DATASET_LABELS:
        return DATASET_LABELS[filename]
    stem = Path(filename).stem
    for suffix in ("chunks_export", "chunks", "export"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return (stem.strip("._-") or stem).upper()


def _asset(name: str) -> str:
    return str(Path(__file__).resolve().parent.parent / "assets" / name)


# Shown when a node in the architecture diagram is clicked.
MERMAID_NODE_NOTES: dict[str, str] = {
    "PDF": "The source electronic datasheet dropped into the pipeline.",
    "PART": "<code>unstructured.partition_pdf(strategy=\"hi_res\")</code> walks the "
    "PDF and returns typed elements instead of one flat string.",
    "TEXT": "Narrative text, including text recovered by OCR from scanned regions.",
    "TABLES": "Electrical-characteristics and recommended-operating tables, kept as "
    "HTML so column headers survive.",
    "IMAGES": "Figures, schematics, pinout diagrams and characteristic curves, kept "
    "as base64 so they can travel with the chunk.",
    "CHUNK": "<code>chunk_by_title</code> groups elements into semantic chunks of at "
    "most 3000 characters.",
    "SEP": "Splits each chunk by content type so text, tables and images can be "
    "handled by the right model.",
    "OCR": "Tesseract OCR reads the text printed inside images: pin labels, axis "
    "values, curve annotations.",
    "VLM": "<b>GLM-4.5V</b> reads the visual content and writes a summary, turning a "
    "figure into something searchable.",
    "ENH": "<code>enhanced_content</code>: the text-only retrieval handle for "
    "knowledge that was originally visual.",
    "CHROMA": "Every chunk is embedded with Granite and persisted with cosine "
    "similarity over HNSW.",
    "JSONF": "The same chunks exported as JSON &mdash; this is what the Chunk "
    "Inspector and Dataset Analytics read.",
    "EMB": "The query is embedded with the same Granite model used at ingestion.",
    "MMR": "Maximal Marginal Relevance picks 10 chunks from a 20-chunk candidate "
    "pool, balancing relevance against redundancy.",
    "PROMPT": "The retrieved chunks are assembled into one multimodal context: text, "
    "tables and the real images.",
    "LLM": "Any configured provider and model, with a bring-your-own API key.",
    "ANSWER": "A grounded answer that cites the chunks it was built from.",
    "EXPLORE": "Open any chunk and inspect raw text, AI summary, images, tables and "
    "metadata, with a 0-100 health score.",
    "ANALYTICS": "Corpus-level distributions and top-20 lists over every chunk.",
    "CHAT": "Ask a question, watch retrieval and the streamed answer, then read the "
    "token, latency and cost stats.",
    "TRACE": "LangSmith traces every embedding, retrieval, context-assembly and "
    "generation call end to end.",
    "EVAL": "A judge model scores retrieval and generation, surfacing failures with "
    "a root cause.",
}


# Full-system architecture, rendered client-side with mermaid.js inside a
# sandboxed iframe.  Kept as data (not a .md file) so the landing page and the
# README always describe the same pipeline.
MERMAID_DIAGRAM = """
flowchart LR
    subgraph Input["PDF Datasheet"]
        PDF[("datasheet.pdf")]
    end

    subgraph EX["1 - Extraction"]
        direction TB
        PART["unstructured partition_pdf<br/>strategy = hi_res"]
        TEXT["raw_text : str"]
        TABLES["tables_html : list[str]"]
        IMAGES["images_base64 : list[str]"]
    end

    subgraph CH["2 - Chunking"]
        CHUNK["chunk_by_title<br/>max_chars = 3000"]
    end

    subgraph EN["3 - Enrichment"]
        SEP["separate_content_types"]
        OCR["Tesseract OCR<br/>text inside images"]
        VLM["GLM-4.5V vision LLM"]
        ENH["enhanced_content : str"]
    end

    subgraph ST["4 - Storage"]
        CHROMA[("ChromaDB<br/>cosine similarity")]
        JSONF[("chunks_huggingface.json")]
    end

    subgraph RE["5 - Retrieval"]
        EMB["Granite embeddings"]
        MMR["MMR retriever<br/>k = 10, fetch_k = 20"]
    end

    subgraph GE["6 - Generation"]
        PROMPT["Multimodal prompt<br/>text + tables + images"]
        LLM["Gemini / any provider<br/>BYO API key"]
        ANSWER["Grounded answer"]
    end

    subgraph UI["7 - Apps and LLMOps"]
        EXPLORE["Chunk Inspector"]
        ANALYTICS["Dataset Analytics"]
        CHAT["Query & Retrieve"]
        TRACE["LangSmith traces"]
        EVAL["LLM-as-a-Judge evals"]
    end

    PDF --> PART
    PART --> TEXT
    PART --> TABLES
    PART --> IMAGES
    PART --> CHUNK
    CHUNK --> SEP
    TEXT --> SEP
    TABLES --> SEP
    IMAGES --> SEP
    SEP --> OCR
    OCR --> VLM
    VLM --> ENH
    ENH --> CHROMA
    ENH --> JSONF
    CHROMA --> EMB
    EMB --> MMR
    MMR --> PROMPT
    PROMPT --> LLM
    LLM --> ANSWER
    JSONF --> EXPLORE
    JSONF --> ANALYTICS
    CHROMA --> CHAT
    CHAT --> ANSWER
    MMR -. traced .-> TRACE
    LLM -. traced .-> TRACE
    CHAT -. traced .-> TRACE
    ANSWER --> EVAL
    EXPLORE --> EVAL

    classDef ingest fill:#1a73e8,color:#fff,stroke:#0b4fa8
    classDef chunk fill:#ea4335,color:#fff,stroke:#a8231a
    classDef enrich fill:#fbbc04,color:#000,stroke:#a07c00
    classDef store fill:#34a853,color:#fff,stroke:#1d6b32
    classDef retrieve fill:#673ab7,color:#fff,stroke:#3f1d80
    classDef ui fill:#ff6d01,color:#fff,stroke:#b34d00
    class PDF,PART,TEXT,TABLES,IMAGES ingest
    class CHUNK chunk
    class SEP,OCR,VLM,ENH enrich
    class CHROMA,JSONF store
    class EMB,MMR,PROMPT,LLM,ANSWER retrieve
    class EXPLORE,ANALYTICS,CHAT,TRACE,EVAL ui
"""


_MERMAID_VIEWER = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<style>
  :root {
    --bg: #ffffff; --panel: #f7f8fa; --fg: #1c1c1c; --muted: #6b7280;
    --line: #e3e6ec; --accent: #ff4b4b;
  }
  html.dark {
    --bg: #0e1117; --panel: #171c26; --fg: #e8eaed; --muted: #98a2b3;
    --line: #272d38; --accent: #ff6b6b;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: transparent; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  #bar { display: flex; align-items: center; gap: 6px; padding: 2px 0 8px 0; }
  #bar button {
    font: inherit; font-size: 12px; font-weight: 600; line-height: 1;
    color: var(--fg); background: var(--panel); cursor: pointer;
    border: 1px solid var(--line); border-radius: 7px; padding: 6px 10px;
  }
  #bar button:hover { border-color: var(--accent); color: var(--accent); }
  #hint { margin-left: 4px; font-size: 11px; color: var(--muted); }
  #zoom {
    margin-left: auto; font-size: 11px; color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  #viewport {
    position: relative; height: 660px; overflow: hidden; cursor: grab;
    background: var(--bg); border: 1px solid var(--line); border-radius: 12px;
  }
  #viewport.dragging { cursor: grabbing; }
  #canvas {
    position: absolute; top: 0; left: 0; width: max-content;
    transform-origin: 0 0;
  }
  /* Mermaid lays the graph out against the width of this element.  Left to
     shrink-to-fit it collapses, every label wraps one word per line and the
     drawing comes out thousands of pixels tall, so give it a real canvas;
     the script below then matches it to the frame. */
  #canvas .mermaid { display: block; width: 1150px; }
  #canvas svg { max-width: none !important; height: auto !important; }
  #canvas g.node { cursor: pointer; }
  #canvas g.mmr-sel > rect, #canvas g.mmr-sel > polygon,
  #canvas g.mmr-sel > circle, #canvas g.mmr-sel > path {
    stroke: var(--accent) !important; stroke-width: 3px !important;
  }
  #note {
    position: absolute; left: 12px; right: 12px; bottom: 12px; display: none;
    padding: 10px 12px; border-radius: 10px; font-size: 12px; line-height: 1.55;
    color: var(--fg); background: var(--panel); border: 1px solid var(--line);
    box-shadow: 0 6px 20px rgba(0,0,0,.18);
  }
  #note.on { display: block; cursor: pointer; }
  #note b { color: var(--accent); }
  #note code { font-size: 11px; }
</style>
</head>
<body>
<div id="bar">
  <button id="zoomOut" title="Zoom out">&minus;</button>
  <button id="zoomIn" title="Zoom in">+</button>
  <button id="fit" title="Fit the whole diagram">Fit</button>
  <button id="actual" title="Render at 100%">1:1</button>
  <span id="hint">drag to pan &middot; double-click to fit &middot; click a node</span>
  <span id="zoom">100%</span>
</div>
<div id="viewport">
  <div id="canvas"><div class="mermaid">
__DIAGRAM__
  </div></div>
  <div id="note"></div>
</div>
<script>
(function () {
  var NOTES = __NOTES__;
  var vp = document.getElementById("viewport");
  var canvas = document.getElementById("canvas");
  var zoomLabel = document.getElementById("zoom");
  var note = document.getElementById("note");
  var svg = null, scale = 1, tx = 0, ty = 0, moved = false;
  var nat = { w: 0, h: 0 };
  var MIN = 0.05, MAX = 8;

  function isDark() {
    try {
      var w = window.parent, d = w.document;
      var els = [
        d.querySelector('[data-testid="stAppViewContainer"]'),
        d.querySelector(".stApp"),
        d.body,
        d.documentElement
      ];
      for (var i = 0; i < els.length; i++) {
        if (!els[i]) continue;
        var c = w.getComputedStyle(els[i]).backgroundColor || "";
        var m = c.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)(?:,\\s*([\\d.]+))?\\)/);
        if (!m) continue;
        if (m[4] !== undefined && parseFloat(m[4]) === 0) continue;
        return (0.299 * m[1] + 0.587 * m[2] + 0.114 * m[3]) < 128;
      }
    } catch (e) { /* parent not reachable: assume the light theme */ }
    return false;
  }

  var dark = isDark();
  if (dark) document.documentElement.classList.add("dark");

  function apply() {
    canvas.style.transform =
      "translate(" + tx + "px," + ty + "px) scale(" + scale + ")";
    zoomLabel.textContent = Math.round(scale * 100) + "%";
  }

  function natural() {
    return nat;
  }

  /* Pin the SVG to its intrinsic pixel size.  Mermaid emits width="100%" plus a
     max-width style, which resolves against the container and can collapse the
     drawing; forcing the viewBox size makes the scale factor predictable. */
  function measure() {
    if (!svg) return { w: 0, h: 0 };
    var vb = svg.viewBox && svg.viewBox.baseVal;
    var w = vb && vb.width, h = vb && vb.height;
    if (!w || !h) {
      var r = svg.getBoundingClientRect();
      w = r.width / scale;
      h = r.height / scale;
    }
    if (w > 0 && h > 0) {
      svg.removeAttribute("width");
      svg.removeAttribute("height");
      svg.style.width = w + "px";
      svg.style.height = h + "px";
    }
    nat = { w: w, h: h };
    return nat;
  }

  function center(s) {
    var r = vp.getBoundingClientRect();
    var n = natural();
    tx = (r.width - n.w * s) / 2;
    ty = (r.height - n.h * s) / 2;
    scale = s;
    apply();
  }

  function fit() {
    var r = vp.getBoundingClientRect();
    var n = natural();
    if (!n.w || !n.h) return;
    center(Math.min((r.width - 24) / n.w, (r.height - 24) / n.h, MAX));
  }

  function zoomBy(factor) {
    var next = Math.min(MAX, Math.max(MIN, scale * factor));
    var r = vp.getBoundingClientRect();
    var cx = r.width / 2, cy = r.height / 2;
    tx = cx - (cx - tx) * (next / scale);
    ty = cy - (cy - ty) * (next / scale);
    scale = next;
    apply();
  }

  function select(g) {
    var prev = canvas.querySelector("g.mmr-sel");
    if (prev) prev.classList.remove("mmr-sel");
    if (!g) { note.classList.remove("on"); return; }
    g.classList.add("mmr-sel");
    var id = (g.id || "").split("-")[1] || "";
    var text = NOTES[id];
    if (!text) { note.classList.remove("on"); return; }
    note.innerHTML = "<b>" + id + "</b> &mdash; " + text;
    note.classList.add("on");
  }

  function bind() {
    document.getElementById("zoomIn").onclick = function () { zoomBy(1.25); };
    document.getElementById("zoomOut").onclick = function () { zoomBy(0.8); };
    document.getElementById("fit").onclick = fit;
    document.getElementById("actual").onclick = function () { center(1); };
    note.onclick = function () { select(null); };

    var drag = false, sx = 0, sy = 0, ox = 0, oy = 0;
    vp.addEventListener("pointerdown", function (e) {
      drag = true; moved = false;
      sx = e.clientX; sy = e.clientY; ox = tx; oy = ty;
      vp.classList.add("dragging");
      try { vp.setPointerCapture(e.pointerId); } catch (err) {}
    });
    vp.addEventListener("pointermove", function (e) {
      if (!drag) return;
      var dx = e.clientX - sx, dy = e.clientY - sy;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) moved = true;
      tx = ox + dx; ty = oy + dy;
      apply();
    });
    ["pointerup", "pointercancel", "pointerleave"].forEach(function (ev) {
      vp.addEventListener(ev, function () {
        drag = false; vp.classList.remove("dragging");
      });
    });
    vp.addEventListener("dblclick", fit);

    Array.prototype.forEach.call(canvas.querySelectorAll("g.node"), function (g) {
      g.addEventListener("click", function () { if (!moved) select(g); });
    });
    window.addEventListener("resize", fit);
  }

  function start() {
    svg = canvas.querySelector("svg");
    bind();

    /* Mermaid finishes measuring labels asynchronously, so the intrinsic size
       can change a frame or two later.  Re-measure, then open framed. */
    function place() {
      measure();
      fit();
    }
    requestAnimationFrame(place);
    setTimeout(place, 120);
  }

  if (window.mermaid) {
    /* Draw the graph at the width of the frame so the intrinsic size of the
       diagram matches the space available, instead of sprawling into a strip
       that then has to be shrunk. */
    var box = document.querySelector(".mermaid");
    if (box) {
      var avail = vp.getBoundingClientRect().width;
      if (avail > 320) box.style.width = avail + "px";
    }

    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: "base",
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      themeVariables: dark ? {
        fontSize: "15px",
        background: "#0e1117",
        primaryColor: "#1b2130",
        primaryTextColor: "#e8eaed",
        lineColor: "#7d8797",
        clusterBkg: "#151a24",
        clusterBorder: "#2a3140",
        edgeLabelBackground: "#0e1117"
      } : {
        fontSize: "15px",
        background: "#ffffff",
        primaryColor: "#f4f6f9",
        primaryTextColor: "#1c1c1c",
        lineColor: "#8a94a6",
        clusterBkg: "#fbfcfd",
        clusterBorder: "#dfe4ec",
        edgeLabelBackground: "#ffffff"
      },
      flowchart: {
        htmlLabels: true, curve: "basis", nodeSpacing: 34, rankSpacing: 46
      }
    });
    window.mermaid.run({ querySelector: ".mermaid" })
      .then(start)
      .catch(function () { start(); });
  }
})();
</script>
</body>
</html>
"""


def _mermaid_iframe(diagram: str) -> str:
    """Build the interactive mermaid viewer document.

    The SVG is rendered at its natural size and scaled with a CSS transform
    inside a fixed viewport, so the diagram fills the frame instead of being
    shrunk by ``max-width``.  Zoom, fit, drag-to-pan and clickable nodes are all
    handled in the iframe; the theme is inherited from the Streamlit app.
    """
    return (
        _MERMAID_VIEWER.replace("__DIAGRAM__", diagram.strip()).replace(
            "__NOTES__", json.dumps(MERMAID_NODE_NOTES)
        )
    )


def log_tracing_status() -> None:
    """Log whether LangSmith tracing is enabled in this runtime.

    ``@traceable`` silently becomes a no-op when tracing is not configured, so
    the app still runs but produces no traces.  These lines make the Cloud
    startup logs show exactly what the SDK sees.
    """
    flag = os.getenv("LANGSMITH_TRACING_V2") or os.getenv("LANGCHAIN_TRACING_V2")
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    project = (
        os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "default"
    )
    background = os.getenv("LANGSMITH_TRACING_BACKGROUND", "true")

    logger.info(
        "LangSmith tracing: enabled=%s, api_key_set=%s, project=%r, "
        "background_post=%s",
        bool(flag and str(flag).lower() == "true"),
        bool(api_key),
        project,
        background,
    )
    if not (flag and str(flag).lower() == "true") or not api_key:
        logger.warning(
            "LangSmith tracing appears DISABLED. Set LANGSMITH_TRACING_V2=true "
            "and LANGSMITH_API_KEY (or LANGCHAIN_* equivalents) in Streamlit "
            "secrets, then confirm the project name matches the one you view."
        )


def init_state() -> None:
    defaults = {
        "data": None,
        "file_name": None,
        "filtered_indices": None,
        "selected_chunk_id": None,
        "search_query": "",
        "has_images": False,
        "has_tables": False,
        "has_raw_text": False,
        "has_enhanced": False,
        "chunk_id_filter": "",
        "min_len": 0,
        "max_len": 100_000,
        "compare_a": None,
        "compare_b": None,
        "chat_query": "",
        "chat_answer": None,
        "chat_retrieval": None,
        "chat_usage": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_data
def load_json(file: Any) -> list[dict]:
    return json.load(file)


def load_json_from_path(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files() -> list[str]:
    json_dir = Path(__file__).resolve().parent.parent / "json"
    if not json_dir.exists():
        return []
    return sorted(f.name for f in json_dir.glob("*.json"))


def _oc(meta: dict) -> dict:
    return meta.get("original_content") or {}


def get_images_base64(chunk: dict) -> list[str]:
    return _oc(chunk.get("metadata", {})).get("images_base64") or []


def get_tables_html(chunk: dict) -> list[str]:
    return _oc(chunk.get("metadata", {})).get("tables_html") or []


def get_raw_text(chunk: dict) -> str:
    return _oc(chunk.get("metadata", {})).get("raw_text") or ""


def get_enhanced_content(chunk: dict) -> str:
    return chunk.get("enhanced_content") or ""


def word_count(text: str) -> int:
    return len(text.split())


def health_score(chunk: dict) -> tuple[int, str]:
    score = 0
    if get_enhanced_content(chunk):
        score += 25
    if get_raw_text(chunk):
        score += 25
    if get_images_base64(chunk):
        score += 25
    if get_tables_html(chunk):
        score += 25

    if score >= 75:
        label = "Pass"
    elif score >= 50:
        label = "Partial"
    else:
        label = "Fail"
    return score, label


def render_image_from_base64(b64_str: str) -> Optional[Image.Image]:
    try:
        raw = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(raw))
    except Exception:
        return None


def parse_table_html(html: str) -> Optional[pd.DataFrame]:
    try:
        tables = pd.read_html(io.StringIO(html))
        if tables:
            return tables[0]
    except Exception:
        pass
    return None


def apply_filters(data: list[dict]) -> list[dict]:
    result = list(data)

    q = st.session_state.search_query.strip().lower()
    if q:
        filtered = []
        for c in result:
            haystack = (
                get_enhanced_content(c) + " " + get_raw_text(c)
            ).lower()
            if q in haystack:
                filtered.append(c)
        result = filtered

    if st.session_state.has_images:
        result = [c for c in result if get_images_base64(c)]
    if st.session_state.has_tables:
        result = [c for c in result if get_tables_html(c)]
    if st.session_state.has_raw_text:
        result = [c for c in result if get_raw_text(c)]
    if st.session_state.has_enhanced:
        result = [c for c in result if get_enhanced_content(c)]

    cid = st.session_state.chunk_id_filter.strip()
    if cid:
        try:
            num = int(cid)
            result = [c for c in result if c.get("chunk_id") == num]
        except ValueError:
            pass

    min_l = st.session_state.min_len
    max_l = st.session_state.max_len
    result = [
        c for c in result if min_l <= len(get_raw_text(c)) <= max_l
    ]

    return result


def _on_filter_change() -> None:
    st.session_state.selected_chunk_id = None
    st.session_state.filtered_indices = None


def _parse_thinking(text: str) -> tuple[str, str]:
    """Extract <think>...</think> blocks from text.

    Returns (thinking_text, answer_text).
    """
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, text, re.DOTALL)
    thinking = "\n\n".join(m.strip() for m in matches)
    answer = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return thinking, answer


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Data")

        json_files = list_json_files()
        if json_files:
            selected_file = st.selectbox(
                "Select a datasheet",
                options=[""] + json_files,
                format_func=lambda x: "Choose a file..." if x == "" else x,
                key="json_selectbox",
            )
            if selected_file:
                json_path = Path(__file__).resolve().parent.parent / "json" / selected_file
                if selected_file != st.session_state.get("file_name"):
                    st.session_state.data = load_json_from_path(str(json_path))
                    st.session_state.file_name = selected_file
                    st.session_state.filtered_indices = None
                    st.session_state.selected_chunk_id = None
                    st.rerun()

        st.divider()
        st.caption("Or upload your own JSON")
        uploaded = st.file_uploader(
            "Upload JSON", type=["json"], label_visibility="collapsed"
        )

        if uploaded and uploaded.name != st.session_state.get("file_name"):
            st.session_state.data = load_json(uploaded)
            st.session_state.file_name = uploaded.name
            st.session_state.filtered_indices = None
            st.session_state.selected_chunk_id = None
            st.rerun()

        data = st.session_state.data
        if data is not None:
            st.metric("Total Chunks", len(data))
            st.divider()

            st.text_input(
                "Search",
                key="search_query",
                placeholder="Search enhanced / raw text ...",
                on_change=_on_filter_change,
            )

            st.divider()
            st.subheader("Filters")

            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Has Images", key="has_images", on_change=_on_filter_change)
                st.checkbox("Has Tables", key="has_tables", on_change=_on_filter_change)
            with col2:
                st.checkbox("Has Raw Text", key="has_raw_text", on_change=_on_filter_change)
                st.checkbox("Has Enhanced", key="has_enhanced", on_change=_on_filter_change)

            st.text_input("Chunk ID", key="chunk_id_filter", on_change=_on_filter_change)

            st.number_input(
                "Min raw text length",
                min_value=0,
                max_value=100_000,
                value=0,
                key="min_len",
                on_change=_on_filter_change,
            )
            st.number_input(
                "Max raw text length",
                min_value=0,
                max_value=100_000,
                value=100_000,
                key="max_len",
                on_change=_on_filter_change,
            )

        # ── LLM Provider (always visible) ──
        st.divider()
        st.subheader("LLM Provider")

        provider_ids = [p.id for p in ALL_PROVIDERS]
        provider_names = {p.id: p.display_name for p in ALL_PROVIDERS}

        selected_provider = st.selectbox(
            "Provider",
            options=provider_ids,
            format_func=lambda x: provider_names.get(x, x),
            key="llm_provider",
            help="Select LLM provider. API key required for custom queries.",
        )

        if selected_provider:
            models = list_models_for_provider(selected_provider)
            model_ids = [m.id for m in models]
            model_names = {m.id: m.display_name for m in models}

            st.selectbox(
                "Model",
                options=model_ids,
                format_func=lambda x: model_names.get(x, x),
                key="llm_model",
                help="Select specific model from the provider",
            )
            st.text_input(
                "Custom Model ID",
                key="llm_custom_model",
                placeholder="e.g. qwen/qwen3.8-27b (overrides dropdown)",
                help="Paste any model ID available on the provider. Leave empty to use dropdown.",
            )

            st.text_input(
                "API Key",
                type="password",
                key="llm_api_key",
                placeholder=f"Enter your {provider_names[selected_provider]} API key",
                help="Your API key for the selected provider. Not stored.",
            )

            st.number_input(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                key="llm_temperature",
            )
            st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=8192,
                value=512,
                key="llm_max_tokens",
            )
            st.number_input(
                "Max Images",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
                key="llm_max_images",
                help="Max images to send to the model. 0 = all.",
            )
            st.number_input(
                "Retrieved Chunks",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
                key="llm_top_k",
                help="How many chunks to retrieve from the vector store.",
            )

            st.checkbox(
                "Use Custom Provider",
                key="use_custom_provider",
                help="When checked, queries use the selected provider/model/key instead of the default backend",
            )
            st.checkbox(
                "Thinking Mode",
                key="llm_thinking",
                help="Ask the model to show its reasoning. Thinking is displayed separately from the answer.",
            )


def render_dashboard(data: list[dict]) -> None:
    total = len(data)
    total_images = sum(len(get_images_base64(c)) for c in data)
    total_tables = sum(len(get_tables_html(c)) for c in data)
    chunks_with_images = sum(1 for c in data if get_images_base64(c))
    chunks_with_tables = sum(1 for c in data if get_tables_html(c))
    avg_raw = (
        sum(len(get_raw_text(c)) for c in data) / total if total else 0
    )
    avg_enh = (
        sum(len(get_enhanced_content(c)) for c in data) / total if total else 0
    )

    cols = st.columns(7)
    metrics = [
        ("Total Chunks", total),
        ("Total Images", total_images),
        ("Total Tables", total_tables),
        ("Chunks w/ Images", chunks_with_images),
        ("Chunks w/ Tables", chunks_with_tables),
        ("Avg Raw Length", f"{avg_raw:.0f}"),
        ("Avg Enhanced Length", f"{avg_enh:.0f}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def build_chunk_df(data: list[dict]) -> pd.DataFrame:
    rows = []
    for c in data:
        cid = c.get("chunk_id", "?")
        n_img = len(get_images_base64(c))
        n_tbl = len(get_tables_html(c))
        rlen = len(get_raw_text(c))
        elen = len(get_enhanced_content(c))
        score, _ = health_score(c)
        rows.append(
            {
                "Chunk ID": cid,
                "Images": n_img,
                "Tables": n_tbl,
                "Raw Len": rlen,
                "Enh Len": elen,
                "Health": score,
            }
        )
    return pd.DataFrame(rows)


def render_chunk_list(data: list[dict]) -> None:
    st.subheader("Chunks")
    df = build_chunk_df(data)
    selection = st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Chunk ID": st.column_config.NumberColumn(width="small"),
            "Images": st.column_config.NumberColumn(width="small"),
            "Tables": st.column_config.NumberColumn(width="small"),
            "Raw Len": st.column_config.NumberColumn(width="medium"),
            "Enh Len": st.column_config.NumberColumn(width="medium"),
            "Health": st.column_config.ProgressColumn(
                format="$score / 100",
                min_value=0,
                max_value=100,
                width="small",
            ),
        },
        on_select="rerun",
        selection_mode="single-row",
    )
    if selection and selection.selection.rows:
        idx = selection.selection.rows[0]
        st.session_state.selected_chunk_id = data[idx].get("chunk_id")


def _health_color(score: int) -> str:
    if score >= 75:
        return "green"
    elif score >= 50:
        return "orange"
    return "red"


def _status_badge(present: bool, label: str) -> str:
    if present:
        return f":green[PASS] {label}"
    return f":red[FAIL] {label}"


def render_detail(chunk: dict) -> None:
    cid = chunk.get("chunk_id", "?")
    images = get_images_base64(chunk)
    tables = get_tables_html(chunk)
    raw = get_raw_text(chunk)
    enhanced = get_enhanced_content(chunk)

    score, label = health_score(chunk)
    color = _health_color(score)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Chunk ID", cid)
    col2.metric("Images", len(images))
    col3.metric("Tables", len(tables))
    col4.metric("Raw Chars", len(raw))
    col5.metric("Enhanced Chars", len(enhanced))

    st.markdown(f"### Health Score: {score}/100  --  :{color}[{label}]")
    st.progress(score / 100)

    qc1, qc2, qc3, qc4 = st.columns(4)
    qc1.markdown(_status_badge(bool(enhanced), "Enhanced Content"))
    qc2.markdown(_status_badge(bool(raw), "Raw Text"))
    qc3.markdown(_status_badge(bool(images), "Images"))
    qc4.markdown(_status_badge(bool(tables), "Tables"))

    col_e1, col_e2, _ = st.columns([1, 1, 6])
    with col_e1:
        st.download_button(
            "Export JSON",
            data=json.dumps(chunk, indent=2),
            file_name=f"chunk_{cid}.json",
            mime="application/json",
        )
    with col_e2:
        export_txt = (
            f"Chunk {cid}\n\n"
            f"--- Enhanced ---\n{enhanced}\n\n"
            f"--- Raw ---\n{raw}"
        )
        st.download_button(
            "Export TXT",
            data=export_txt,
            file_name=f"chunk_{cid}.txt",
            mime="text/plain",
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Enhanced Content", "Raw Text", "Images", "Tables", "Metadata"]
    )

    with tab1:
        _render_enhanced_tab(enhanced)
    with tab2:
        _render_raw_tab(raw)
    with tab3:
        _render_images_tab(images)
    with tab4:
        _render_tables_tab(tables)
    with tab5:
        _render_metadata_tab(chunk)


def _render_enhanced_tab(text: str) -> None:
    st.markdown(text)

    wc = word_count(text)
    cc = len(text)

    col1, col2 = st.columns(2)
    col1.metric("Words", wc)
    col2.metric("Characters", cc)

    with st.expander("Show / Hide"):
        st.markdown(text)


def _render_raw_tab(text: str) -> None:
    wc = word_count(text)
    cc = len(text)

    col1, col2 = st.columns(2)
    col1.metric("Words", wc)
    col2.metric("Characters", cc)

    st.text_area(
        "Raw Text",
        value=text,
        height=400,
        disabled=True,
        label_visibility="collapsed",
    )

    st.download_button(
        "Download", data=text, file_name="raw_text.txt", mime="text/plain"
    )


def _render_images_tab(images: list[str]) -> None:
    if not images:
        st.info("No images found in this chunk.")
        return

    for i, b64 in enumerate(images, 1):
        st.subheader(f"Image {i}")
        img = render_image_from_base64(b64)
        if img is None:
            st.warning(f"Image {i} could not be decoded (malformed base64).")
            continue

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        bts = buf.getvalue()

        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(img, width="stretch")
        with col2:
            st.write(f"**Width:** {img.width} px")
            st.write(f"**Height:** {img.height} px")
            st.write(f"**Size:** {len(bts) / 1024:.1f} KB")
            st.write(f"**Format:** {img.format or 'N/A'}")

            st.download_button(
                "Download",
                data=bts,
                file_name=f"chunk_image_{i}.png",
                mime="image/png",
                key=f"dl_img_{i}",
            )

        with st.expander(f"Zoom Image {i}"):
            st.image(img, width=800)


def _render_tables_tab(tables: list[str]) -> None:
    if not tables:
        st.info("No tables found in this chunk.")
        return

    for i, html in enumerate(tables, 1):
        st.subheader(f"Table {i}")
        df = parse_table_html(html)
        if df is not None:
            st.dataframe(df, width="stretch")
        else:
            st.warning(
                f"Could not parse Table {i} as DataFrame. Showing raw HTML."
            )
            st.html(html)

        with st.expander(f"Raw HTML -- Table {i}"):
            st.code(html, language="html")


def _render_metadata_tab(chunk: dict) -> None:
    meta = chunk.get("metadata", {})
    images = get_images_base64(chunk)
    tables = get_tables_html(chunk)

    col1, col2, col3 = st.columns(3)
    col1.metric("Image Count", len(images))
    col2.metric("Table Count", len(tables))
    col3.write(f"**Metadata keys:** {list(meta.keys())}")

    st.divider()
    st.json(meta)


def render_compare(data: list[dict]) -> None:
    st.header("Compare Chunks")

    ids = [c.get("chunk_id") for c in data]

    col_a, col_b = st.columns(2)
    with col_a:
        a_id = st.selectbox("Chunk A", ids, key="cmp_a")
    with col_b:
        b_id = st.selectbox("Chunk B", ids, key="cmp_b")

    chunk_a = next((c for c in data if c.get("chunk_id") == a_id), None)
    chunk_b = next((c for c in data if c.get("chunk_id") == b_id), None)

    if not chunk_a or not chunk_b:
        st.warning("Select two chunks to compare.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Chunk {a_id}")
        _render_mini_chunk(chunk_a)
    with col2:
        st.subheader(f"Chunk {b_id}")
        _render_mini_chunk(chunk_b)


def _render_mini_chunk(chunk: dict) -> None:
    images = get_images_base64(chunk)
    tables = get_tables_html(chunk)
    raw = get_raw_text(chunk)
    enhanced = get_enhanced_content(chunk)
    score, label = health_score(chunk)
    color = _health_color(score)

    st.metric("Health Score", f"{score}/100")
    st.progress(score / 100)
    st.write(f"**Images:** {len(images)}  |  **Tables:** {len(tables)}")
    st.write(
        f"**Raw:** {len(raw)} chars  |  **Enhanced:** {len(enhanced)} chars"
    )

    with st.expander("Enhanced Content"):
        st.markdown(enhanced or "*empty*")
    with st.expander("Raw Text"):
        st.text(raw or "*empty*")
    with st.expander(f"Images ({len(images)})"):
        for i, b64 in enumerate(images, 1):
            img = render_image_from_base64(b64)
            if img:
                st.image(img, width=300, caption=f"Image {i}")
    with st.expander(f"Tables ({len(tables)})"):
        for i, html in enumerate(tables, 1):
            df = parse_table_html(html)
            if df is not None:
                st.dataframe(df)
            else:
                st.html(html)
    with st.expander("Metadata"):
        st.json(chunk.get("metadata", {}))


def render_analytics(data: list[dict]) -> None:
    st.header("Dataset Analytics")

    df = pd.DataFrame(
        [
            {
                "chunk_id": c.get("chunk_id"),
                "images": len(get_images_base64(c)),
                "tables": len(get_tables_html(c)),
                "raw_len": len(get_raw_text(c)),
                "enhanced_len": len(get_enhanced_content(c)),
            }
            for c in data
        ]
    )

    if df.empty:
        st.info("No data to analyse.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Images", df["images"].sum())
    m2.metric("Total Tables", df["tables"].sum())
    m3.metric("Avg Chunk Size (chars)", f"{df['raw_len'].mean():.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(
            df,
            x="images",
            title="Distribution of Images per Chunk",
            labels={"images": "Image Count", "count": "Chunks"},
            nbins=df["images"].max() + 1,
        )
        st.plotly_chart(fig1, width="stretch")
    with col2:
        fig2 = px.histogram(
            df,
            x="tables",
            title="Distribution of Tables per Chunk",
            labels={"tables": "Table Count", "count": "Chunks"},
            nbins=df["tables"].max() + 1,
        )
        st.plotly_chart(fig2, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(
            df,
            x="raw_len",
            title="Distribution of Raw Text Length",
            labels={"raw_len": "Character Count", "count": "Chunks"},
        )
        st.plotly_chart(fig3, width="stretch")
    with col4:
        fig4 = px.histogram(
            df,
            x="enhanced_len",
            title="Distribution of Enhanced Content Length",
            labels={"enhanced_len": "Character Count", "count": "Chunks"},
        )
        st.plotly_chart(fig4, width="stretch")

    st.subheader("Top 20 Lists")
    tcol1, tcol2, tcol3 = st.columns(3)

    with tcol1:
        st.markdown("**Largest Chunks (raw text)**")
        top_raw = df.nlargest(20, "raw_len")[["chunk_id", "raw_len"]]
        st.dataframe(top_raw, width="stretch")

    with tcol2:
        st.markdown("**Most Images**")
        top_img = df.nlargest(20, "images")[["chunk_id", "images"]]
        st.dataframe(top_img, width="stretch")

    with tcol3:
        st.markdown("**Most Tables**")
        top_tbl = df.nlargest(20, "tables")[["chunk_id", "tables"]]
        st.dataframe(top_tbl, width="stretch")


def page_explorer(data: list[dict]) -> None:
    render_dashboard(data)

    filtered = apply_filters(data)
    st.info(f"Showing {len(filtered)} of {len(data)} chunks")

    if not filtered:
        st.warning("No chunks match the current filters.")
        return

    list_col, detail_col = st.columns([1, 1.8])

    with list_col:
        render_chunk_list(filtered)
        cid_filter = st.session_state.chunk_id_filter.strip()
        if not cid_filter:
            sel = st.selectbox(
                "Select chunk",
                options=[c.get("chunk_id") for c in filtered],
                format_func=lambda x: f"Chunk {x}",
                key="chunk_selector",
                label_visibility="collapsed",
                index=None,
                placeholder="Pick a chunk...",
            )
            if sel is not None:
                st.session_state.selected_chunk_id = sel
                st.rerun()

    with detail_col:
        selected_id = st.session_state.selected_chunk_id
        if selected_id is not None:
            chunk = next(
                (c for c in data if c.get("chunk_id") == selected_id), None
            )
            if chunk:
                with st.container(border=True):
                    render_detail(chunk)
            else:
                st.info("Select a chunk from the list.")


# ---------------------------------------------------------------------------
# Query & Retrieve page (talks to the FastAPI backend)
# ---------------------------------------------------------------------------
def _render_chunk_payload(payload: dict, index: int) -> None:
    """Render one retrieved chunk from an API ``ChunkPayload`` dict."""
    raw_text = payload.get("raw_text", "")
    tables = payload.get("tables_html", [])
    image_urls = payload.get("image_urls", [])
    enhanced = payload.get("enhanced_content", "")

    label = (
        f"Chunk {index + 1}  —  "
        f"Images: {len(image_urls)}  |  "
        f"Tables: {len(tables)}  |  "
        f"Raw: {len(raw_text)} chars  |  "
        f"Enhanced: {len(enhanced)} chars"
    )

    with st.expander(label, expanded=index == 0):
        tab_e, tab_r, tab_i, tab_t = st.tabs(["Enhanced", "Raw Text", "Images", "Tables"])

        with tab_e:
            st.markdown(enhanced)

        with tab_r:
            st.metric("Words", len(raw_text.split()))
            st.text_area(
                "raw",
                value=raw_text,
                height=250,
                disabled=True,
                label_visibility="collapsed",
                key=f"api_raw_{index}",
            )

        with tab_i:
            if not image_urls:
                st.info("No images in this chunk.")
            else:
                for j, url in enumerate(image_urls):
                    st.image(url, width=400, caption=f"Image {j + 1}")

        with tab_t:
            if not tables:
                st.info("No tables in this chunk.")
            else:
                for j, html in enumerate(tables):
                    st.markdown(f"**Table {j + 1}**")
                    df = parse_table_html(html)
                    if df is not None:
                        st.dataframe(df, width="stretch")
                    else:
                        st.warning("Could not parse table.")
                        st.code(html, language="html")


def _render_retrieval(retrieval: dict) -> None:
    """Render the ``chunks`` of a retrieve/query API response."""
    chunks = retrieval.get("chunks", [])
    with st.expander(f"**Referenced Chunks** — {len(chunks)} chunks", expanded=True):
        for i, payload in enumerate(chunks):
            _render_chunk_payload(payload, i)


def _render_usage(usage: Optional[dict]) -> None:
    """Render token usage and generation stats returned by the backend."""
    if not usage:
        return

    st.markdown("#### Usage")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Model", usage.get("model", "—"))
    col_b.metric("Input tokens", usage.get("input_tokens", 0))
    col_c.metric("Output tokens", usage.get("output_tokens", 0))
    col_d.metric("Total tokens", usage.get("total_tokens", 0))

    latency = usage.get("latency_ms")
    latency_label = f"{latency:,.0f} ms" if isinstance(latency, (int, float)) else "—"
    cost = usage.get("estimated_cost_usd")
    cost_label = f"${cost:.6f}" if isinstance(cost, (int, float)) else "—"

    col_e, col_f, col_g, col_h = st.columns(4)
    col_e.metric("Latency", latency_label)
    col_f.metric("Prompt", f"{usage.get('prompt_chars', 0)} chars")
    col_g.metric("Images", usage.get("prompt_images", 0))
    col_h.metric("Est. cost", cost_label)


def render_chat_page() -> None:
    st.header("Query & Retrieve")

    st.markdown(
        "Enter a query to search the vector store and generate "
        "a multimodal answer with referenced chunks. All retrieval and "
        "generation runs on the FastAPI backend."
    )

    # Check the backend is reachable before doing anything else.
    try:
        backend_status = api_health()
        backend_ok = backend_status.get("status") == "ok"
    except Exception:
        backend_ok = False

    if not backend_ok:
        st.error(
            f"Backend API unreachable at `{API_BASE_URL}`. "
            f"Start it with `uvicorn api.main:app --reload --port 8000`."
        )
    else:
        st.success(f"Backend connected — vector store: {backend_status.get('vector_store')}")

    query = st.text_area(
        "Query",
        value=st.session_state.chat_query,
        placeholder="Ask a question about your documents ...",
        key="chat_query_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submitted = st.button("Search", type="primary", width="stretch")
    with col2:
        clear = st.button("Clear", width="stretch")
        if clear:
            st.session_state.chat_answer = None
            st.session_state.chat_retrieval = None
            st.session_state.chat_usage = None
            st.rerun()

    if submitted and query.strip():
        st.session_state.chat_query = query
        try:
            use_custom = st.session_state.get("use_custom_provider", False)

            if use_custom:
                # Custom provider mode - use /query/custom endpoint
                provider = st.session_state.get("llm_provider")
                model = st.session_state.get("llm_model")
                custom_model = (st.session_state.get("llm_custom_model") or "").strip()
                if custom_model:
                    model = custom_model
                api_key = st.session_state.get("llm_api_key")
                temperature = st.session_state.get("llm_temperature", 0.0)
                max_tokens = st.session_state.get("llm_max_tokens", 512)
                max_images = st.session_state.get("llm_max_images", 0)
                thinking = st.session_state.get("llm_thinking", False)

                if not (provider and model and api_key):
                    st.error("Please select provider, model, and enter API key for custom provider mode.")
                    return

                status = st.status(f"Querying {provider}/{model} ...", expanded=True)
                answer_placeholder = st.empty()

                with st.spinner("Retrieving chunks and generating answer..."):
                    result = query_custom(
                        query=query,
                        top_k=st.session_state.get("llm_top_k", 10),
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_images=max_images,
                        thinking=thinking,
                    )

                retrieval = result
                chunks = retrieval.get("chunks", [])
                st.session_state.chat_retrieval = retrieval
                status.update(
                    label=f"Retrieved {len(chunks)} chunks",
                    state="complete",
                    expanded=False,
                )
                _render_retrieval(retrieval)

                answer = result.get("answer", "")
                # Parse <think> tags from the answer
                thinking_text, answer_text = _parse_thinking(answer)
                if thinking_text:
                    with st.expander("Thinking", expanded=False):
                        st.markdown(thinking_text)
                answer_placeholder.markdown(f"### Answer\n\n{answer_text}")
                st.session_state.chat_answer = answer
                st.session_state.chat_usage = result.get("usage")
                _render_usage(result.get("usage"))

            else:
                # Default backend mode - use streaming
                events = api_stream(query, top_k=st.session_state.get("llm_top_k", 10))
                status = st.status("Retrieving relevant chunks ...", expanded=True)
                answer_placeholder = st.empty()
                collected: list[str] = []
                thinking_collected: list[str] = []
                in_thinking = False

                for event, data in events:
                    if event == "retrieval":
                        retrieval = data
                        chunks = retrieval.get("chunks", [])
                        st.session_state.chat_retrieval = retrieval
                        status.update(
                            label=f"Retrieved {len(chunks)} chunks",
                            state="complete",
                            expanded=False,
                        )
                        _render_retrieval(retrieval)
                    elif event == "token":
                        delta = data.get("delta", "")
                        # Parse <think> tags dynamically
                        for char in delta:
                            collected.append(char)
                            full = "".join(collected)
                            # Detect state transitions
                            if not in_thinking and "<think>" in full and "</think>" not in full:
                                in_thinking = True
                                thinking_collected = []
                            if in_thinking:
                                thinking_text = "".join(thinking_collected)
                                # Check if we just got the closing tag
                                if "</think>" in thinking_text + char:
                                    in_thinking = False
                                    thinking_collected = []
                                else:
                                    thinking_collected.append(char)
                        # Show answer (without <think> tags) with cursor
                        _, answer_so_far = _parse_thinking("".join(collected))
                        answer_placeholder.markdown(f"### Answer\n\n{answer_so_far}▌")
                    elif event == "done":
                        full_answer = data.get("answer", "")
                        thinking_text, answer_text = _parse_thinking(full_answer)
                        if thinking_text:
                            with st.expander("Thinking", expanded=False):
                                st.markdown(thinking_text)
                        answer_placeholder.markdown(f"### Answer\n\n{answer_text}")
                        st.session_state.chat_answer = full_answer
                        st.session_state.chat_usage = data.get("usage")
                        _render_usage(data.get("usage"))
                    elif event == "error":
                        status.update(label="Answer generation failed", state="error")
                        st.error(data.get("message", "Unknown backend error"))

                if not collected:
                    answer_placeholder.markdown("### Answer\n\n*No answer returned.*")
        except Exception as e:
            st.error(f"Query failed: {e}")

    # Display persisted results on subsequent script runs.
    answer = st.session_state.get("chat_answer")
    retrieval = st.session_state.get("chat_retrieval")

    if answer is None and retrieval is None:
        st.info("Enter a query and press Search to get started.")
        return

    if submitted and query.strip():
        return

    if retrieval:
        _render_retrieval(retrieval)

    if answer:
        st.markdown("### Answer")
        st.markdown(answer)

    _render_usage(st.session_state.get("chat_usage"))


# ---------------------------------------------------------------------------
# Landing page -- full project walkthrough, shown when no datasheet is loaded
# ---------------------------------------------------------------------------
def _render_hero() -> None:
    buttons = "".join(
        f'<a class="mmr-btn" href="{url}" target="_blank" rel="noopener">'
        f"{label} &#8599;</a>"
        for label, url in PROJECT_LINKS
    )
    badges = "".join(
        f'<span class="mmr-badge">{tech}</span>' for tech in CORE_STACK
    )

    st.markdown(
        f"""
        <div class="mmr-hero">
          <h2>Multimodal RAG AI Pipeline for Electronic Datasheets</h2>
          <p>A production pipeline that reads datasheets the way an engineer does:
          narrative text, tables, figures, schematics, pinout diagrams and the
          relationships between them &mdash; then answers questions grounded in
          the exact chunks and images that support them.</p>
          <div class="mmr-actions">{buttons}</div>
          <div class="mmr-badges">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_problem() -> None:
    st.markdown("## The problem with text-only RAG on datasheets")
    st.markdown(
        """
A datasheet rarely states its most important facts in sentences. The answer
lives in the visuals: pin assignments drawn on a package outline, absolute
maximum ratings inside a table, gain and dropout curves, switching waveforms,
typical application circuits. A text-only pipeline retrieves the paragraph
*near* the figure and calls it a day, so the model confidently answers the wrong
part &mdash; or nothing at all.

This project treats every modality as first-class: extraction, enrichment,
storage, retrieval and generation all carry text, tables **and** images together.
"""
    )


def _html_grid(fragments: list[str], css_class: str) -> str:
    """Join HTML fragments into one raw-HTML block safe for markdown.

    Fragments are stripped before joining: markdown ends a raw HTML block on the
    first blank line, and a whitespace-only line counts as blank.  Leaving the
    indentation that trails each triple-quoted fragment in place would escape
    every card after the first one into visible markup.
    """
    body = "".join(f.strip() for f in fragments)
    return f'<div class="{css_class}">{body}</div>'


def _render_pipeline_steps() -> None:
    st.markdown("## What we built &mdash; step by step")

    cards = []
    for step, title, body, tags in PIPELINE_STEPS:
        tag_html = "".join(f'<span class="mmr-tag">{t}</span>' for t in tags)
        cards.append(
            f"""<div class="mmr-card">
  <div class="mmr-step">{step}</div>
  <h4>{title}</h4>
  <p>{body}</p>
  <div class="mmr-tags">{tag_html}</div>
</div>"""
        )

    st.markdown(_html_grid(cards, "mmr-grid"), unsafe_allow_html=True)


def _render_results() -> None:
    st.markdown("## Evaluation results")
    st.markdown(
        f"""
The [LLM-as-a-Judge framework]({EVAL_DASHBOARD_URL}) grades retrieval and
generation separately across **13 curated datasheet questions**. Headline
numbers from the live run:
"""
    )

    cells = []
    for value, label, hint in EVAL_METRICS:
        cells.append(
            f"""<div class="mmr-metric">
  <div class="v">{value}</div>
  <div class="l">{label}</div>
  <div class="h">{hint}</div>
</div>"""
        )

    st.markdown(_html_grid(cells, "mmr-metrics"), unsafe_allow_html=True)
    st.caption(
        "Open the Evaluation Dashboard for the per-question score distribution, "
        "the retrieval-vs-generation comparison and the failure analysis that "
        "explains every non-pass case."
    )


def _render_architecture() -> None:
    st.markdown("## Architecture")
    st.markdown(
        "End-to-end multimodal data flow &mdash; PDF in, grounded answer out, "
        "with every stage traced."
    )

    diagram_html = _mermaid_iframe(MERMAID_DIAGRAM)
    if hasattr(st, "iframe"):
        st.iframe(diagram_html, height=710)
    else:  # Streamlit < 1.45, where st.iframe does not exist yet
        import streamlit.components.v1 as components

        components.html(diagram_html, height=710)

    with st.expander("Mermaid source"):
        st.code(MERMAID_DIAGRAM.strip(), language="mermaid")


def _render_how_to_explore() -> None:
    st.markdown("## How to explore this project")
    st.markdown(
        "Pick a datasheet in the **sidebar** (or upload your own chunk-export JSON) "
        "and every page below becomes live."
    )

    steps = [
        (
            "Select a datasheet",
            "Sidebar &rarr; *Select a datasheet*. Five datasets ship with the repo: "
            + ", ".join(f"**{_dataset_display_name(f)}**" for f in list_json_files())
            + ". The sidebar then reports total chunks and unlocks the filters.",
        ),
        (
            "Chunk Explorer",
            "Dataset-wide metrics, then filter by search text, images, tables, raw "
            "text, enhanced content or chunk ID. Click any row to open that chunk's "
            "health score, AI summary, raw OCR text, decoded images, parsed tables "
            "and full metadata.",
        ),
        (
            "Compare Chunks",
            "Put any two chunks side by side &mdash; useful to see how one visual "
            "region got summarised and how it differs from a text-only chunk.",
        ),
        (
            "Dataset Analytics",
            "Distributions of images, tables, raw length and enhanced length, plus "
            "top-20 lists for largest chunks and most multimodal chunks.",
        ),
        (
            "Query &amp; Retrieve",
            "Ask a question and watch the backend retrieve the top-10 chunks and "
            "stream a grounded answer. Expand any retrieved chunk to see the exact "
            "text, tables and images the model was given, then check token usage, "
            "latency and estimated cost. Choose any provider, model or Thinking Mode "
            "in the sidebar &mdash; or enable <i>Use Custom Provider</i> and bring "
            "your own API key.",
        ),
        (
            "Evaluation Dashboard",
            f"The judge results live in a [separate deployment]({EVAL_DASHBOARD_URL}): "
            "per-question scores, retrieval vs generation breakdown, metric averages "
            "and root-cause diagnosis for failures.",
        ),
    ]

    st.markdown(
        "\n\n".join(
            f"{i}. **{title}** &mdash; {body}"
            for i, (title, body) in enumerate(steps, 1)
        )
    )


def _render_gallery() -> None:
    shots = [
        ("Chunk Explorer", "dashboard-chunk-explorer.png"),
        ("Query & Retrieve", "dashboard-query.png"),
        ("Dataset Analytics", "dashboard-analytics.png"),
        ("Compare Chunks", "dashboard-compare.png"),
    ]
    available = [(label, _asset(f)) for label, f in shots if Path(_asset(f)).exists()]
    if not available:
        return

    with st.expander("Screenshots &mdash; the app in action"):
        for label, path in available:
            st.image(path, caption=label, width="stretch")


def render_landing_page() -> None:
    """Project walkthrough shown until a datasheet is selected."""
    _render_hero()
    st.divider()
    _render_problem()
    st.divider()
    _render_pipeline_steps()
    st.divider()
    _render_architecture()
    st.divider()
    _render_results()
    st.divider()
    _render_how_to_explore()
    st.divider()
    _render_gallery()

    st.markdown(
        f"""
        <div class="mmr-hero">
          <h2>Ready to look under the hood?</h2>
          <p>Select a datasheet in the sidebar to inspect real chunks, or jump
          straight to Query &amp; Retrieve to ask the pipeline a question.</p>
          <div class="mmr-actions">
            <a class="mmr-btn" href="{GITHUB_URL}" target="_blank" rel="noopener">Source code &#8599;</a>
            <a class="mmr-btn" href="{EVAL_DASHBOARD_URL}" target="_blank" rel="noopener">Evaluation Dashboard &#8599;</a>
            <a class="mmr-btn" href="{LIVE_APP_URL}" target="_blank" rel="noopener">Live App &#8599;</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.75rem; }
        div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th {
            font-size: 0.8rem;
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        hr { margin: 0.5rem 0; }
        h1 { font-size: 1.6rem; margin-bottom: 0; }
        h2 { font-size: 1.2rem; }
        h3 { font-size: 1.0rem; }

        /* ── Landing page ── */
        .mmr-hero {
            border: 1px solid rgba(128,128,128,.28);
            border-left: 4px solid #ff4b4b;
            border-radius: 14px;
            padding: 22px 26px;
            background: linear-gradient(120deg,
                rgba(255,75,75,.10), rgba(99,102,241,.10));
        }
        .mmr-hero h2 { margin: 0 0 8px 0; font-size: 1.45rem; line-height: 1.25; }
        .mmr-hero p { margin: 0; font-size: 0.95rem; line-height: 1.6; opacity: 0.85; }
        .mmr-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
        .mmr-btn {
            display: inline-block; padding: 7px 14px; border-radius: 999px;
            font-size: 0.82rem; font-weight: 600; text-decoration: none;
            border: 1px solid rgba(128,128,128,.38); color: inherit;
            transition: border-color .15s, color .15s;
        }
        .mmr-btn:hover { border-color: #ff4b4b; color: #ff4b4b; }
        .mmr-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 16px; }
        .mmr-badge {
            font-size: 0.72rem; padding: 3px 10px; border-radius: 6px;
            background: rgba(128,128,128,.14);
            border: 1px solid rgba(128,128,128,.22); color: inherit;
        }
        .mmr-grid {
            display: grid; gap: 12px;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        }
        .mmr-card {
            border: 1px solid rgba(128,128,128,.22); border-radius: 12px;
            padding: 16px 18px; background: rgba(128,128,128,.04);
        }
        .mmr-step {
            font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
            text-transform: uppercase; color: #ff4b4b;
        }
        .mmr-card h4 { margin: 5px 0 8px 0; font-size: 1rem; line-height: 1.35; }
        .mmr-card p { margin: 0; font-size: 0.85rem; line-height: 1.6; opacity: 0.8; }
        .mmr-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 12px; }
        .mmr-tag {
            font-size: 0.68rem; padding: 2px 8px; border-radius: 5px;
            background: rgba(128,128,128,.12);
            border: 1px solid rgba(128,128,128,.18); color: inherit; opacity: 0.9;
        }
        .mmr-metrics {
            display: grid; gap: 12px;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        }
        .mmr-metric {
            text-align: center; padding: 16px 12px; border-radius: 12px;
            border: 1px solid rgba(128,128,128,.22); background: rgba(128,128,128,.04);
        }
        .mmr-metric .v { font-size: 1.6rem; font-weight: 700; line-height: 1.1; }
        .mmr-metric .l {
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
            margin-top: 6px; opacity: 0.75;
        }
        .mmr-metric .h { font-size: 0.7rem; margin-top: 6px; opacity: 0.6; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(PAGE_TITLE)
    init_state()
    render_sidebar()

    data = st.session_state.data

    page = st.sidebar.radio(
        "Navigation",
        [
            "Chunk Explorer",
            "Compare Chunks",
            "Dataset Analytics",
            "Query & Retrieve",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    if st.session_state.file_name:
        st.sidebar.caption(
            f"Loaded: {st.session_state.file_name}  "
            f"-  Chunks: {len(data)}"
        )

    if page == "Query & Retrieve":
        render_chat_page()
    elif data is None:
        render_landing_page()
    elif page == "Chunk Explorer":
        page_explorer(data)
    elif page == "Compare Chunks":
        render_compare(data)
    elif page == "Dataset Analytics":
        render_analytics(data)


if __name__ == "__main__":
    log_tracing_status()
    main()
