# BFSI Compliance Document Assistant

A locally-run RAG (Retrieval-Augmented Generation) system for querying BFSI regulatory documents — built as an implementation blueprint for AI deployment in regulated environments where data residency, auditability, and answer verification are non-negotiable.


## What This Project Demonstrates

This was built as a hands-on exploration of AI implementation constraints specific to regulated industries — not as a generic RAG tutorial. The questions it was built to answer:

- What does "data never leaves the machine" actually require, architecturally?
- What does a confidence score need to measure to be trustworthy, not just present?
- Where does an LLM's output need a human-verifiable anchor, and how do you build the UI to make that anchor easy to reach, not optional?
- What is the actual latency cost structure of CPU-only local inference, and which levers genuinely move it?

**Everything runs on a local CPU-only machine. Zero API cost. Zero data leaves the device.**

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   PDF Doc   │ ──▶ │   Chunking   │ ──▶ │  Embedding   │ ──▶ │   ChromaDB   │
│  (PyMuPDF)  │     │ (paragraph + │     │ (all-MiniLM- │     │  (local,     │
│             │     │   overlap)   │     │   L6-v2)     │     │  persistent) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Streamlit  │ ◀── │  Confidence  │ ◀── │   Phi-3      │ ◀── │   Query +    │
│     UI      │     │   Scoring    │     │   Mini       │     │   Top-K      │
│             │     │ (dual-layer) │     │  (Ollama)    │     │   Retrieval  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

**Stack:** Python · Streamlit · ChromaDB · sentence-transformers · Ollama (Phi-3 Mini) · PyMuPDF

---

## Key Design Decisions

### Why local inference (Ollama + Phi-3 Mini), not an API

Regulatory documents — KYC records, AML policy, credit files — cannot leave client infrastructure in most BFSI engagements. Running inference locally via Ollama eliminates this constraint entirely: no document text, query, or generated answer ever crosses the network boundary. Phi-3 Mini (3.7GB) was selected specifically because it runs on CPU-only hardware within a 15GB RAM constraint — the kind of machine a compliance team would realistically be issued, not a GPU workstation.

### Why source chunks are always shown alongside the answer

This is the single most important design decision in the project, and it came directly from a debugging finding (see Learnings below): the LLM can state a fact correctly while citing the wrong source for it. A generated answer is not a substitute for the source document — it is a navigation aid to it. The UI never lets a user see an answer without also being one click away from the raw text that produced it.

### Why confidence scoring is dual-layer, not single-metric

Vector distance alone is an unreliable confidence signal. A query can retrieve a *semantically related but factually insufficient* chunk — meaning the distance score is good even though the LLM has nothing to actually answer with. The system runs two independent checks and treats either one failing as low confidence:

1. **Retrieval confidence** — is the best-matching chunk's cosine distance below threshold (0.55)?
2. **Generation confidence** — does the LLM's own answer contain a "could not find" signal?

```python
low_confidence = (best_distance > CONFIDENCE_THRESHOLD) or answer_indicates_not_found
```

### Why temperature=0

Compliance Q&A has no room for creative variation. Two identical queries should produce the same answer. Default sampling temperature introduced answer instability — the same query sometimes returned a complete, correct answer and sometimes an incomplete one. Setting `temperature=0` made generation deterministic and was the single highest-leverage fix in the project for answer reliability.

### Why paragraph-level chunking with overlap, not fixed-size chunking

Regulatory documents are structured around numbered clauses with sub-clauses (a, b, c...) that depend on shared context (an `Explanation:` note, a parent clause heading). Naive fixed-size chunking splits these mid-definition. Paragraph-boundary chunking with overlap keeps related sub-clauses together more often, at the cost of variable chunk size — a deliberate trade-off favoring semantic integrity over uniform chunk length.

---

## What Was Measured, Not Assumed

Every performance claim below came from instrumented pipeline runs, not estimation.

| Stage | Typical latency | % of total |
|-------|-----------------|------------|
| Embed query | 25–50 ms | <0.1% |
| Retrieve (ChromaDB) | 2–60 ms | <0.1% |
| LLM generation | 7s – 165s | ~99.9% |

**Retrieval is effectively free. Generation is the entire latency story.** This single finding redirected all subsequent optimization effort away from the vector pipeline and onto prompt size and inference behavior.

---

## Learnings

These are documented because each one changed the architecture — they are not a list of obstacles, they are the actual design history of the project.

### 1. A correct fact can carry an incorrect citation

On a query about beneficial ownership thresholds in trusts, the model correctly stated the 10% interest threshold — but attributed it to the wrong regulatory amendment date, conflating two adjacent footnote references in the source chunk. The fact was right; the citation was hallucinated. This is the finding that justified always surfacing raw source chunks in the UI rather than trusting LLM-generated citations.

### 2. KV-cache behavior, not "cold start," explains most latency variance

Initial latency spikes (90–165s) were assumed to be model cold-start. Instrumented testing showed otherwise: Ollama/llama.cpp reuses cached key-value state for the **prefix** of a prompt shared with the previously processed prompt. An exact-repeat query is fast (~7–12s, decode-only) because the entire prompt matches the cache. A *rephrased* version of the same question lands in between (~22s) — partial prefix match. A genuinely novel query pays full prefill cost, which scales roughly linearly with prompt size (~25ms per character observed). **In real usage, every user query is novel — so the fast repeat-query numbers are not representative of production latency.** The actionable finding: reducing `TOP_K` and `chunk_size` reduces prompt size, which is the only lever that meaningfully reduces latency on CPU-only inference. Output length tuning (`num_predict`) has comparatively minor effect, because prefill — not decode — dominates the cost.

### 3. Prompt instructions calibrated for one failure mode can cause another

An instruction to "extract ALL explicitly stated limits, don't stop at the first one" fixed incomplete answers on multi-part numeric queries (e.g., a three-part account limit). The same instruction caused the model to over-extrapolate on dense single-sentence legal definitions — inventing an unstated "50% ownership" threshold that did not exist in the source. The fix was adding an explicit, separate instruction prohibiting inference beyond what's stated, alongside the completeness instruction — they are not the same constraint and need to be stated independently.

### 4. Output truncation looks like a parsing bug but is a token-budget problem

Structured JSON outputs (used in a companion agent built on this pipeline) intermittently failed to parse. The cause was not malformed JSON — it was the model running out of `num_predict` tokens mid-object on verbose inputs. The fix was raising the token budget *and* constraining the model to shorter field values in the prompt itself, plus adding a truncation-recovery parser as a fallback (close any open brace/bracket structures before giving up).

---

## Known Limitations

- **Latency is not production-grade for live chat.** 80–165s per novel query on CPU is appropriate for a back-office compliance lookup tool, not a real-time customer-facing assistant. GPU inference or a hosted model would be the production fix.
- **No PII detection layer is currently wired into the pipeline.** Microsoft Presidio was evaluated for query-side PII screening (PAN, Aadhaar, phone number patterns) but is not yet integrated into `app.py`.
- **No audit logging is currently active in this version.** A CSV-based audit trail (timestamp, query, retrieved chunks, confidence, latency) was prototyped but removed from the shipped app in favor of keeping the UI layer minimal; the `ask()` function already returns every field needed to log this if re-added.
- **Chunking is not regulatory-clause-aware.** It approximates clause boundaries via paragraph splitting; it does not parse the document's actual clause/sub-clause numbering structure.

