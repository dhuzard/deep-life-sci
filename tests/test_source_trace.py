"""Run-scoped provenance around source-tool calls."""

from langchain_core.tools import tool

from deep_life_sci.middleware.source_trace import research_trace, with_source_trace


@tool
async def fetch_abstracts(pmids: list[str]) -> dict:
    """Small stand-in for PubMed abstract retrieval."""
    return {
        "records": {pmid: {"pmid": pmid, "abstract": "payload"} for pmid in pmids},
        "missing": [],
        "invalid": [],
        "from_cache": [],
    }


async def test_trace_keeps_identifiers_but_not_scientific_payloads():
    wrapped = with_source_trace([fetch_abstracts])[0]
    with research_trace() as trace:
        result = await wrapped.ainvoke({"pmids": ["123", "456"]})

    assert result["records"]["123"]["abstract"] == "payload"
    data = trace.as_dict()
    assert data["pmids"] == ["123", "456"]
    assert data["events"][0]["requested_ids"] == ["123", "456"]
    assert "payload" not in repr(data)
