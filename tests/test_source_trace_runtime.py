"""Integration coverage for provenance across the real QuickJS tool bridge."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deep_life_sci import agent, runner
from deep_life_sci.sandbox import ResilientSandbox


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


async def test_runner_trace_survives_quickjs_source_bridge(monkeypatch, mock_ncbi):
    model = ScriptedModel(
        responses=[
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "eval",
                        "id": "search",
                        "args": {
                            "code": 'const r = await tools.pubmedSearch('
                            '{term: "cancer", retmax: 0}); '
                            'console.log("COUNT=" + r.count);'
                        },
                    }
                ],
            ),
            AIMessage("Found 7 papers."),
        ]
    )
    monkeypatch.setattr(agent, "root_model", lambda: model)
    monkeypatch.setattr(agent, "subagent_model", lambda: model)
    monkeypatch.setattr(agent, "register_harness_profile", Mock())
    transport = mock_ncbi(
        lambda req: httpx.Response(
            200,
            json={
                "esearchresult": {
                    "count": "7",
                    "idlist": [],
                    "querytranslation": "cancer",
                }
            },
        )
    )

    backend = ResilientSandbox(SimpleNamespace())
    monkeypatch.setattr(
        backend, "aexecute", AsyncMock(return_value=SimpleNamespace(output=""))
    )

    result = await runner.run_once(
        "Count cancer papers",
        backend=backend,
    )

    assert result.answer == "Found 7 papers."
    assert len(transport.requests) == 1
    assert result.as_dict()["source_trace"] == result.source_trace
    assert result.source_trace["pmids"] == []
    assert len(result.source_trace["events"]) == 1
    event = result.source_trace["events"][0]
    assert event["source"] == "pubmed"
    assert event["tool"] == "pubmed_search"
    assert event["operation"] == "search"
    assert event["query"] == "cancer"
    assert event["success"] is True
