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
