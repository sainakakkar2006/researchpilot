from conftest import BAD_REVIEW, GOOD_REVIEW, FakeLLM, cite

from researchpilot.config import Settings
from researchpilot.orchestrator import Orchestrator
from researchpilot.schemas import AgentEvent

PLAN = [[{"text": "SQ one?"}]]
GOOD_EV = ("Grounded answer.", [cite()], ["query"])
BAD_EV = ("Shaky answer.", [], [])


def run(llm, **settings_kw):
    events: list[AgentEvent] = []
    settings = Settings(**settings_kw)
    report = Orchestrator(llm, settings, on_event=events.append).run("Main question?")
    return report, events


def test_happy_path_accepts_first_attempt():
    llm = FakeLLM(json_q=PLAN, grounded_q=[GOOD_EV], tools_q=[GOOD_REVIEW],
                  text_q=["Final answer."])
    report, events = run(llm)
    assert report.summary == "Final answer."
    assert report.groundedness_score >= 0.7
    assert report.corrective_passes == 0
    assert llm.calls_made == 4  # plan + research + critique + synthesize
    assert any(e.action == "accept" for e in events)
    assert not any(e.action == "retry" for e in events)


def test_self_correction_retry_on_failed_factcheck():
    llm = FakeLLM(json_q=PLAN,
                  grounded_q=[BAD_EV, GOOD_EV],          # attempt 1 bad, attempt 2 good
                  tools_q=[BAD_REVIEW, GOOD_REVIEW],
                  text_q=["Final answer."])
    report, events = run(llm)
    retries = [e for e in events if e.action == "retry"]
    assert len(retries) == 1
    # critic's refined query is fed back into the researcher
    assert "official GDP statistics 2026" in retries[0].detail
    assert report.evidence[0].attempt == 2
    assert report.evidence[0].grounded


def test_keeps_best_attempt_when_retries_exhausted():
    llm = FakeLLM(json_q=PLAN,
                  grounded_q=[BAD_EV, BAD_EV],
                  tools_q=[BAD_REVIEW, [("finish_review", {"overall_confidence": 0.4})]],
                  text_q=["Final answer."],
                  max_api_calls=6)  # no headroom for a corrective pass
    report, events = run(llm, max_research_retries=1)
    # second attempt (0.4 capped to 0.3 ungrounded... actually kept as best)
    assert report.evidence  # still produced a report from the best attempt
    assert any("keeping best attempt" in e.detail for e in events)
    assert report.caveats  # unsupported claim surfaced as caveat


def test_report_level_corrective_pass():
    # First pass: ungrounded + low confidence -> groundedness < 0.6 -> corrective
    llm = FakeLLM(json_q=PLAN,
                  grounded_q=[BAD_EV, BAD_EV, BAD_EV, GOOD_EV],
                  tools_q=[BAD_REVIEW,
                           [("finish_review", {"overall_confidence": 0.2})],
                           BAD_REVIEW, GOOD_REVIEW],
                  text_q=["Draft.", "Corrected final."])
    report, events = run(llm, max_research_retries=1)
    assert report.corrective_passes == 1
    assert report.summary == "Corrected final."
    assert any(e.action == "correct" for e in events)
    assert report.groundedness_score >= 0.7


def test_budget_guard_stops_gracefully():
    llm = FakeLLM(json_q=PLAN, grounded_q=[GOOD_EV], tools_q=[GOOD_REVIEW],
                  text_q=["Final."], max_api_calls=2)  # plan + 1 research only
    report, events = run(llm, max_api_calls=2)
    assert any(e.action == "budget" for e in events)
    # no crash; report still returned
    assert report.question == "Main question?"


def test_event_stream_covers_all_agents():
    llm = FakeLLM(json_q=PLAN, grounded_q=[GOOD_EV], tools_q=[GOOD_REVIEW],
                  text_q=["Final."])
    _, events = run(llm)
    agents = {e.agent for e in events}
    assert {"planner", "researcher", "critic", "synthesizer", "orchestrator"} <= agents
