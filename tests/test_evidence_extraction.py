"""The root prompt's evidence fan-out, executed verbatim across the real QuickJS bridge.

The helpers in the prompt are production code the root model copies into `eval`. These
tests run those exact blocks through `build_agent` with scripted models, so the dynamic
`responseSchema`, the DeepAgents task boundary and the JS-side checks are all real. Only
the models and the NCBI transport are fakes.
"""

import json
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from deep_life_sci import agent, runner
from deep_life_sci.prompts.system import _TEMPLATE
from deep_life_sci.sandbox import ResilientSandbox

_FIELDS = ["status", "finding", "locator", "excerpt", "limitations"]


def _js_block(marker: str) -> str:
    """The one ```js block in the root prompt that contains `marker`."""
    [block] = [b for b in re.findall(r"```js\n(.*?)```", _TEMPLATE, re.S) if marker in b]
    return block


ABSTRACT_FANOUT = _js_block("const evidenceSchema")
FULL_TEXT_FANOUT = _js_block('"full-text-analyst"')
HELPERS = ABSTRACT_FANOUT.split("const { records }")[0]


class ScriptedRoot(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class RoutedLeaf(BaseChatModel):
    """Answers each dispatch by the source marker in its task description.

    A fan-out runs its leaves concurrently, so a list-ordered fake would hand replies to
    papers in scheduling order. Routing on the description keeps each reply on its paper.
    """

    routes: dict[str, Any]
    bound_tool_names: list[list[str]] = Field(default_factory=list)
    descriptions: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "routed-leaf"

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names.append([getattr(t, "name", None) or t["name"] for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        description = messages[-1].text
        self.descriptions.append(description)
        for marker, reply in self.routes.items():
            if marker in description:
                if isinstance(reply, Exception):
                    raise reply
                return ChatResult(generations=[ChatGeneration(message=reply)])
        raise AssertionError(f"no scripted reply for: {description[:80]}")


def _extraction(**args) -> AIMessage:
    """A leaf answering through the structured-output tool the schema title names."""
    return AIMessage(
        "", tool_calls=[{"name": "EvidenceExtraction", "id": "x", "args": args}]
    )


def _abstracts(request: httpx.Request) -> httpx.Response:
    assert request.url.path.endswith("/efetch.fcgi")
    articles = "".join(
        f"""<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>
<ArticleTitle>Paper {pmid}</ArticleTitle><Abstract>
<AbstractText Label="METHODS">BODY-{pmid} {"x" * 2000}</AbstractText>
</Abstract></Article></MedlineCitation></PubmedArticle>"""
        for pmid in request.url.params["id"].split(",")
    )
    return httpx.Response(200, text=f"<PubmedArticleSet>{articles}</PubmedArticleSet>")


def _backend(monkeypatch) -> tuple[ResilientSandbox, dict[str, str]]:
    written: dict[str, str] = {}
    backend = ResilientSandbox(SimpleNamespace())
    monkeypatch.setattr(
        backend, "aexecute", AsyncMock(return_value=SimpleNamespace(output=""))
    )

    async def awrite(path, content):
        written[path] = content
        return SimpleNamespace(error=None, path=path, files_update=None)

    monkeypatch.setattr(backend, "awrite", awrite)
    return backend, written


async def _run(
    monkeypatch, mock_ncbi, *scripts: str, routes: dict[str, Any],
    fetched: list[str] | None = None,
):
    """Run scripted `eval` calls; `fetched` collects the PMIDs the fake NCBI served."""

    def serve(request: httpx.Request) -> httpx.Response:
        if fetched is not None:
            fetched.extend(request.url.params["id"].split(","))
        return _abstracts(request)

    calls = [
        AIMessage("", tool_calls=[{"name": "eval", "id": f"e{i}", "args": {"code": code}}])
        for i, code in enumerate(scripts)
    ]
    root = ScriptedRoot(responses=[*calls, AIMessage("done")])
    leaf = RoutedLeaf(routes=routes)
    monkeypatch.setattr(agent, "root_model", lambda: root)
    monkeypatch.setattr(agent, "subagent_model", lambda: leaf)
    monkeypatch.setattr(agent, "register_harness_profile", Mock())
    mock_ncbi(serve)
    backend, written = _backend(monkeypatch)
    result = await runner.run_once("Did these studies use mice?", backend=backend)
    results = [m.text for m in result.messages if isinstance(m, ToolMessage)]
    return result, results, leaf, written


def _eval_value(tool_text: str) -> Any:
    """Parse an eval result that ended in `JSON.stringify(...)` (strings render raw)."""
    assert "<error" not in tool_text, tool_text
    return json.loads(tool_text.split("<result>", 1)[1].rsplit("</result>", 1)[0])


# Eval results render as JS inspect output, so assertions re-emit the prompt's own
# projection as JSON. The context-size test reads the verbatim result instead.
AS_JSON = "\nJSON.stringify(answers.map(evidenceSummary));"


def _abstract_script(pmids: list[str], *, as_json: bool = True) -> str:
    script = f"const pmids = {json.dumps(pmids)};\n{ABSTRACT_FANOUT}"
    return script + AS_JSON if as_json else script


async def test_each_status_crosses_task_with_fetched_identity(monkeypatch, mock_ncbi):
    routes = {
        "PMID: 101": _extraction(
            status="supported", finding="Mice were used.", locator="METHODS",
            excerpt="BODY-101", limitations="none",
        ),
        "PMID: 102": _extraction(
            status="not_addressed", finding="The abstract does not say.",
            locator="none", excerpt="none", limitations="none",
        ),
        "PMID: 103": _extraction(
            status="insufficient", finding="The abstract is truncated.",
            locator="none", excerpt="none", limitations="Only a fragment was supplied.",
        ),
    }
    fetched: list[str] = []
    _, [tool_text], leaf, written = await _run(
        monkeypatch, mock_ncbi, _abstract_script(["101", "102", "103"]), routes=routes,
        fetched=fetched,
    )

    summary = {row["pmid"]: row for row in _eval_value(tool_text)}
    assert {pmid: row["status"] for pmid, row in summary.items()} == {
        "101": "supported", "102": "not_addressed", "103": "insufficient",
    }
    assert summary["103"]["limitations"] == "Only a fragment was supplied."

    # The schema reached each leaf as its structured-output tool, beside its one file tool.
    assert leaf.bound_tool_names == [["read_file", "EvidenceExtraction"]] * 3

    # The full packets, excerpts included, are in the file.
    [(path, content)] = written.items()
    assert path.endswith("answers.json")
    packets = json.loads(content)
    assert {p["extraction"]["excerpt"] for p in packets} == {"BODY-101", "none"}

    # Every attached identity is exactly the record the source client fetched, one per
    # served PMID, and none of it is drawn from what the model wrote.
    assert sorted(fetched) == ["101", "102", "103"]
    assert sorted(
        (p["source"] for p in packets), key=lambda s: s["pmid"]
    ) == [
        {"kind": "pubmed_abstract", "pmid": pmid, "title": f"Paper {pmid}",
         "retracted": False}
        for pmid in sorted(fetched)
    ]
    for p in packets:
        assert not set(p["source"].values()) & set(p["extraction"].values())
    assert set(summary) == set(fetched)
    assert all(list(p["extraction"]) == _FIELDS for p in packets)


async def test_analyst_cannot_replace_or_add_source_identity(monkeypatch, mock_ncbi):
    """Raw JSON schemas are not validated by ToolStrategy, so extra keys reach JS."""
    routes = {
        "PMID: 101": _extraction(
            status="supported", finding="Reported in PMID 999.", locator="METHODS",
            excerpt="none", limitations="none",
            pmid="999", source={"kind": "pmc_full_text", "pmid": "999"}, pmcid="PMC9",
        ),
    }
    _, [tool_text], _, written = await _run(
        monkeypatch, mock_ncbi, _abstract_script(["101"]), routes=routes
    )

    [row] = _eval_value(tool_text)
    assert row["pmid"] == "101"
    assert "pmcid" not in row
    [packet] = json.loads(next(iter(written.values())))
    assert packet["source"]["pmid"] == "101"
    assert packet["source"]["kind"] == "pubmed_abstract"
    assert list(packet["extraction"]) == _FIELDS


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(
            _extraction(status="maybe", finding="f", locator="l", excerpt="e",
                        limitations="none"),
            id="status-outside-enum",
        ),
        pytest.param(
            _extraction(status="supported", finding="f", locator="l", excerpt="e"),
            id="missing-field",
        ),
        pytest.param(
            _extraction(status="supported", finding=None, locator="l", excerpt="e",
                        limitations="none"),
            id="null-field",
        ),
        pytest.param(AIMessage("Mice were used."), id="free-text-no-tool-call"),
    ],
)
async def test_invalid_extraction_is_an_error_not_evidence(
    monkeypatch, mock_ncbi, reply
):
    routes = {
        "PMID: 101": reply,
        "PMID: 102": _extraction(
            status="supported", finding="Mice were used.", locator="METHODS",
            excerpt="none", limitations="none",
        ),
    }
    _, [tool_text], _, written = await _run(
        monkeypatch, mock_ncbi, _abstract_script(["101", "102"]), routes=routes
    )

    # One bad leaf does not reject the whole Promise.all.
    summary = {row["pmid"]: row for row in _eval_value(tool_text)}
    assert summary["102"]["status"] == "supported"
    assert summary["101"] == {"pmid": "101", "error": "analyst returned no valid extraction"}
    [packet] = [
        p for p in json.loads(next(iter(written.values()))) if p["source"]["pmid"] == "101"
    ]
    assert "extraction" not in packet


async def test_failed_task_keeps_its_error_visible_to_the_root(monkeypatch, mock_ncbi):
    """A rejected `task()` fails the call as before; catching it in JS loses the text."""
    routes = {"PMID: 101": RuntimeError("gateway said no")}
    _, [tool_text], _, written = await _run(
        monkeypatch, mock_ncbi, _abstract_script(["101"]), routes=routes
    )

    assert tool_text.startswith("<error")
    assert "gateway said no" in tool_text
    assert written == {}


async def test_full_text_fanout_reuses_helpers_and_keeps_pmcid(monkeypatch, mock_ncbi):
    """The full-text block depends on helpers an earlier `eval` left in scope."""
    routes = {
        "PMID: 101": _extraction(
            status="supported", finding="f", locator="l", excerpt="e", limitations="none",
        ),
        "PMCID: PMC7": _extraction(
            status="supported", finding="n = 12 mice.", locator="Methods",
            excerpt="twelve C57BL/6 mice", limitations="none",
        ),
    }
    full_text = (
        'const question = "How many mice?";\n'
        'const full = { PMC7: { pmcid: "PMC7", pmid: "107", title: "Paper 7", '
        'retracted: false, text: "FULLTEXT-BODY" } };\n' + FULL_TEXT_FANOUT + AS_JSON
    )
    _, [_, tool_text], leaf, _ = await _run(
        monkeypatch, mock_ncbi, _abstract_script(["101"]), full_text, routes=routes
    )

    assert _eval_value(tool_text) == [{
        "pmid": "107", "pmcid": "PMC7", "status": "supported",
        "finding": "n = 12 mice.", "locator": "Methods", "limitations": "none",
    }]
    assert any("FULLTEXT-BODY" in d for d in leaf.descriptions)


async def test_projection_keeps_source_bodies_and_excerpts_out_of_root(
    monkeypatch, mock_ncbi
):
    routes = {
        f"PMID: {pmid}": _extraction(
            status="supported", finding="Mice were used.", locator="METHODS",
            excerpt=f"EXCERPT-{pmid}", limitations="none",
        )
        for pmid in ("101", "102", "103")
    }
    _, [tool_text], leaf, _ = await _run(
        monkeypatch, mock_ncbi, _abstract_script(["101", "102", "103"], as_json=False),
        routes=routes,
    )

    # Each ~2 kB abstract went to its leaf and nowhere near the root.
    assert sum("BODY-" in d for d in leaf.descriptions) == 3
    assert "BODY-" not in tool_text
    assert "EXCERPT-" not in tool_text
    assert "undefined" not in tool_text
    # Three bounded rows, not three abstracts: well under one abstract's length.
    assert len(tool_text) < 1_000


async def test_schema_follows_task_interface_constraints(monkeypatch, mock_ncbi):
    """Single JSON types, closed shape, and no identifier fields for the model to fill."""
    script = _abstract_script([]) + "\nJSON.stringify(evidenceSchema);"
    _, [tool_text], _, _ = await _run(monkeypatch, mock_ncbi, script, routes={})

    schema = _eval_value(tool_text)
    assert schema["title"] == "EvidenceExtraction"
    assert schema["additionalProperties"] is False
    assert schema["required"] == _FIELDS == list(schema["properties"])
    assert all(p["type"] == "string" for p in schema["properties"].values())
    assert schema["properties"]["status"]["enum"] == [
        "supported", "not_addressed", "insufficient",
    ]
    assert not re.search(r"pmid|pmcid|nct|source", json.dumps(list(schema["properties"])))
