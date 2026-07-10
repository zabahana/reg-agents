# Model Card — GenAI Complaint Classification & Regulatory Mapping (GENAI-COMPLAINT-030)

## Purpose
Classifies inbound consumer complaints and maps them to applicable federal
regulations (e.g., ECOA, FCRA, EFTA/Reg E, UDAAP) to route to the right teams
and surface regulatory risk trends.

## Methodology
Retrieval-augmented generation: complaints embedded and matched against a
regulation knowledge base (NeMo Retriever embeddings + vector search); an LLM
(served via NVIDIA NIM) produces the classification, mapped regulation(s), and
a rationale. A lightweight fine-tuned classifier provides a fast first pass.

## Training / Grounding Data
120k historically labeled complaints mapped across 24 federal regulations;
regulation corpus curated by Regulatory Intelligence and Legal.

## Features / Inputs
Free-text complaint narrative, product/channel metadata, retrieved regulation
passages.

## Performance
Macro-F1 0.88 on held-out complaints; retrieval hit-rate@5 0.93. Human review
required for low-confidence or high-severity categories.

## Known Limitations
- Hallucination risk on ambiguous complaints; mitigated by retrieval grounding
  and confidence thresholds.
- Regulation corpus must be kept current; stale passages degrade mapping.

## Monitoring
Weekly F1 on sampled human-reviewed complaints; retrieval-grounding checks;
prompt-injection and jailbreak red-teaming each release; drift on category mix.

## Documentation Gaps (self-reported)
- Explanation-faithfulness evaluation is qualitative; needs a quantitative
  metric.
- Guardrail coverage for PII leakage needs formal test cases.
