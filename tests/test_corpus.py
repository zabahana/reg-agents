"""Corpus + retrieval tests. Fully offline (lexical fallback)."""

from reg_agents.common.corpus import lexical_search, load_regulations


def test_regulations_load():
    docs = load_regulations()
    assert len(docs) > 5
    sources = {d.metadata["source"] for d in docs}
    assert "sr11-7_model_risk.md" in sources
    assert "ecoa_reg_b.md" in sources


def test_lexical_search_finds_sr11_7():
    docs = load_regulations()
    hits = lexical_search(docs, "effective challenge model validation", k=3)
    assert hits
    joined = " ".join(h.document.text.lower() for h in hits)
    assert "challenge" in joined or "validation" in joined
