# Product research: what to offer someone who uploads a datasheet or PCB document

This note covers the short-term product only. A user uploads a PDF (a datasheet, an application note, a reference design, or a schematic) and asks questions. The goal is to decide which questions and utilities to support, and what would make this project stand out from the tools that already exist.

Research was done in November 2026. Vendor claims are reported as claims, not verified facts.

## What already exists

| Tool | What it does | Notes |
|---|---|---|
| PartGenie Datasheet AI | Upload a PDF or enter a part number; ask about pinouts, thermal limits, replacement risk; cites spec sections | 25 MB upload cap |
| Datasheet Copilot (open source) | Answers only from loaded datasheets with page citations; says so when a value is not stated; unit-checked calculations | Closest to our scope |
| SpecsCore AI | Local Windows app; checks manufacturer PDFs against design limits; citations open a local PDF preview | Pitch is privacy: NDA datasheets never leave the machine |
| Datasheet.chat | Q&A and summaries across several documents | General purpose |
| Flux Copilot | In-editor PCB assistant; reads the datasheet linked to a part; design reviews (unconnected pins, missing decoupling); builds parts from PDFs | Strong on schematics, self-described as weak on layout |
| Traceformer | LLM schematic review from KiCad or Altium exports plus datasheets | Targets datasheet-level mistakes that ERC misses; does not accept PDF schematics |
| DRCY (research) | Multi-agent check that a schematic follows the manufacturer's recommended connections | Runs as a CI step |

Three things are common to all serious tools:

1. **Citations to a page**, because engineers do not trust an unsourced number.
2. **Refusing to guess** when the document does not state a value.
3. **Privacy**, because many datasheets and all internal schematics are confidential.

Hacker News feedback on LLM schematic checkers was mixed. The main criticism is that output can look like analysis without being correct. Proof of accuracy is therefore a differentiator in itself.

## What engineers actually get wrong with datasheets

These are the recurring pitfalls in practitioner writing. Each is a question the product should answer well.

- **Reading only the first page.** Key limits sit deep in the document, in footnotes and test conditions.
- **Typical versus guaranteed.** A "typ" value is a characterisation, not a promise. Min and max values are the guarantee, and only under stated conditions.
- **Absolute maximum versus recommended operating conditions.** The two are often confused.
- **Conditions attached to numbers.** "Dropout 1.1 V" is meaningless without the load current and junction temperature it was measured at.
- **Skipping the graphs.** Performance curves describe real behaviour (dropout versus current, efficiency versus load, derating).
- **Thermal limits.** Power dissipation and θJA are underestimated for regulators and MOSFETs.
- **Layout and decoupling guidance** is in the datasheet, and is often skipped.
- **Sequencing and power-up order** can destroy some parts.
- **Errata and application notes** contain information the datasheet leaves out, and datasheets themselves contain mistakes.

## What we should offer, in priority order

The current corpus is TLV1117, LM317, LM2596, LM555, and the AN699 application note. Every feature below can be demonstrated on that corpus.

### Tier 1: makes the core Q&A trustworthy

1. **Answers cited to page and region.** Every claim names the document, the page, and the table or figure, and the UI shows the cropped region. This requires keeping page numbers and bounding boxes during ingestion (see `docs/TRADEOFFS.md`, section 1).
2. **Spec answers with their conditions.** "Dropout voltage is 1.2 V typical, 1.3 V max, at IOUT = 800 mA, TJ = 25 °C (Table 7.5, page 5)". The prompt must require the min, typ, and max values and the test conditions whenever they exist.
3. **"Not stated in this document"** as a first-class answer, measured in the evaluation set.
4. **Figure and graph questions.** "What does the efficiency-versus-load curve show at 2 A?" This is where a multimodal pipeline beats text-only tools, and it is the reason this project exists. Be honest that reading values off curves is approximate.

### Tier 2: utilities that save real time

5. **A key-specs card on upload.** One structured extraction per document: part number, package, absolute maximum ratings, recommended operating conditions, key electrical characteristics, and the pinout table. Return it as JSON from the API and render it as a card. It is useful on its own, and it makes a strong demo.
6. **Pinout and pin-function questions.** "What does pin 4 of the LM2596 do and what should it connect to?"
7. **Design calculations with the datasheet's formula.** For example:
   - LM317 output voltage: `VOUT = 1.25 V × (1 + R2/R1) + IADJ × R2`;
   - LM2596 inductor and capacitor selection;
   - regulator thermal checks: `PD = (VIN − VOUT) × IOUT`, then `TJ = TA + PD × θJA`.

   The formula is cited from the document and the arithmetic is done in code, never by the model. A small set of hand-written calculators for these parts is enough at this stage.
8. **Layout and decoupling checklist.** Extract the "Layout Guidelines" and "Application Information" sections into a checklist (capacitor values and placement, feedback trace routing, thermal pad, ground return).
9. **Comparing two parts.** "TLV1117 versus LM317 for 3.3 V at 500 mA from 5 V: dropout, quiescent current, thermal headroom." This needs document-scoped retrieval with several documents in one query.

### Tier 3: later, once the above is solid

10. **Schematic PDF against datasheet.** The user uploads their schematic PDF together with the datasheets of the parts on it. The tool checks that each part's recommended connections are followed (feedback divider, enable pin, bypass capacitors). This is what Traceformer and DRCY do from CAD exports. Doing it from a PDF is harder, because the netlist has to be recovered from an image. It is the natural bridge to the long-term PCB agent.
11. **Errata and revision awareness.** Flag when a datasheet revision is old or an errata sheet exists. This needs external data, so it belongs to the agent phase.
12. **Local or private mode.** A local embedder and a local summarizer, so confidential documents never leave the machine. Part of this is already planned through the local vision model option.

## On uploading PCB layouts specifically

A PCB layout PDF (copper layers, silkscreen, and assembly drawings) contains almost no text. Our pipeline indexes text and summaries, so for a layout PDF it would retrieve little that is useful, and the generator would be guessing from a picture. Layout review needs geometry (traces, clearances, layer stack-up). That comes from Gerber, ODB++, or KiCad files, not from a rendered PDF. Even Flux describes its own layout understanding as limited.

**Recommendation for this phase:** accept datasheets, application notes, and reference-design PDFs first. Treat schematic PDFs as Tier 3, and layout files as part of the later agent phase. This is a decision for the owner (see the open questions below).

## Where this project can stand out

Most existing tools are text-first. Few publish how accurate they are. This project's edge is:

1. **Truly multimodal.** It answers from tables and figures, not just paragraphs, and shows the figure it used.
2. **Measured.** A public evaluation set over real datasheets (text, table, and figure questions, plus "not stated" questions) with retrieval and answer accuracy reported for each pipeline change. Academic benchmarks such as UniDoc-Bench and REAL-MM-RAG show that table and figure questions are exactly where RAG systems are weakest, so numbers here mean something.
3. **Engineering-correct.** Conditions attached to every spec, min, typ, and max kept distinct, and calculations done in code.
4. **Explainable architecture.** Every component has a written trade-off (see `docs/TRADEOFFS.md`).

## Open questions for the owner

1. Which documents should the first upload version accept: datasheets and app notes only, or schematic PDFs as well?
2. Is privacy (local models, nothing leaves the machine) a goal for this phase, or later?
3. Who is the target user for the demo: a student learning to read datasheets, or a working engineer checking a design? This changes the tone and the features on the first screen.
4. Should the key-specs card (item 5) be part of the upload flow in Checkpoint 7, or a separate later feature?

## Sources

- [PartGenie Datasheet AI](https://www.partgenie.ai/datasheet-ai)
- [Datasheet Copilot on GitHub](https://github.com/Aatikah-Sheikh/datasheet-copilot)
- [SpecsCore AI announcement](https://www.gaussianwaves.com/2026/10/specscore-ai-local-datasheet-compliance/)
- [Datasheet.chat](https://genai.works/applications/datasheet-chat-lvqyy7vc)
- [Best AI tools for engineering documentation (Leo AI)](https://www.getleo.ai/blog/best-ai-tools-engineering-technical-documentation-2026)
- [Flux Copilot documentation](https://docs.flux.ai/reference/copilot)
- [Flux: AI design reviews](https://prod.flux.ai/p/blog/ai-design-reviews)
- [Traceformer](https://traceformer.io/)
- [Show HN: LLM-powered PCB schematic checker](https://news.ycombinator.com/item?id=46492601)
- [DRCY: Agentic hardware design reviews (arXiv)](https://arxiv.org/pdf/2603.15672)
- [Surveying GenAI-based automation in PCB design and test (arXiv)](https://arxiv.org/pdf/2606.17074)
- [UniDoc-Bench (arXiv)](https://arxiv.org/html/2510.03663v2)
- [REAL-MM-RAG (arXiv)](https://arxiv.org/pdf/2502.12342)
- [Lost in OCR Translation? Vision-based document retrieval (arXiv)](https://arxiv.org/abs/2505.05666)
- [Analog Devices RAQ: reading datasheets](https://www.analog.com/en/resources/analog-dialogue/raqs/raq-issue-79.html)
- [Advanced Circuits: how to read component datasheets](https://www.advancedpcb.com/en-us/resources/blog/how-to-read-electronic-component-datasheets/)
- [Electronic Design: interpreting datasheet data](https://www.electronicdesign.com/technologies/analog/article/21133999/all-that-glitters-is-not-gold-interpreting-datasheet-data-when-selecting-parts)
