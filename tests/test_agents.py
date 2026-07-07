from types import SimpleNamespace

from conftest import BAD_REVIEW, GOOD_REVIEW, FakeLLM, cite

from researchpilot.agents import Critic, Planner, Researcher, Synthesizer
from researchpilot.llm import GeminiClient
from researchpilot.schemas import Critique, Evidence, SubQuestion


# ----------------------------------------------------------------- planner
def test_planner_parses_json_array():
    llm = FakeLLM(json_q=[[{"text": "Q1?", "rationale": "r1"},
                          {"text": "Q2?", "rationale": "r2"}]])
    subs = Planner(llm).plan("Main question?")
    assert [s.text for s in subs] == ["Q1?", "Q2?"]
    assert subs[0].id == 1 and subs[0].rationale == "r1"


def test_planner_unwraps_object_and_caps_count():
    llm = FakeLLM(json_q=[{"sub_questions": [{"text": f"Q{i}?"} for i in range(9)]}])
    subs = Planner(llm, max_sub_questions=3).plan("Main?")
    assert len(subs) == 3


def test_planner_falls_back_to_original_question():
    llm = FakeLLM(json_q=[ValueError("bad json"), ValueError("bad json")])
    subs = Planner(llm).plan("Main question?")
    assert len(subs) == 1
    assert subs[0].text == "Main question?"
    assert subs[0].rationale == "planner fallback"


# -------------------------------------------------------------- researcher
def test_researcher_builds_evidence_with_citations():
    llm = FakeLLM(grounded_q=[("Answer text.", [cite()], ["query one"])])
    ev = Researcher(llm).research(SubQuestion(id=1, text="Q?"))
    assert ev.grounded and ev.answer == "Answer text."
    assert ev.search_queries == ["query one"]


def test_researcher_passes_refinement_hint():
    captured = {}

    class SpyLLM(FakeLLM):
        def generate_grounded(self, prompt):
            captured["prompt"] = prompt
            return super().generate_grounded(prompt)

    llm = SpyLLM(grounded_q=[("A.", [], [])])
    Researcher(llm).research(SubQuestion(id=1, text="Q?"),
                             refined_query="better query", attempt=2)
    assert "better query" in captured["prompt"]
    assert "failed fact-checking" in captured["prompt"]


# ------------------------------------------------------------------ critic
def test_critic_parses_function_calls():
    llm = FakeLLM(tools_q=[BAD_REVIEW])
    ev = Evidence(sub_question_id=1, answer="GDP grew 900%", citations=[cite()])
    cr = Critic(llm).review(SubQuestion(id=1, text="Q?"), ev)
    assert len(cr.unsupported) == 1
    assert cr.overall_confidence == 0.3
    assert cr.refined_query == "official GDP statistics 2026"
    assert not cr.passed


def test_critic_caps_confidence_when_ungrounded():
    llm = FakeLLM(tools_q=[[("finish_review", {"overall_confidence": 0.95})]])
    ev = Evidence(sub_question_id=1, answer="text", citations=[])  # no citations
    cr = Critic(llm).review(SubQuestion(id=1, text="Q?"), ev)
    assert cr.overall_confidence <= 0.3


def test_critic_handles_missing_finish_review():
    llm = FakeLLM(tools_q=[[]])  # model returned no function calls
    ev = Evidence(sub_question_id=1, answer="text", citations=[cite()])
    cr = Critic(llm).review(SubQuestion(id=1, text="Q?"), ev)
    assert cr.overall_confidence == 0.5  # distrust default, grounded


# ------------------------------------------------------------- synthesizer
def test_groundedness_math():
    ev_g = Evidence(sub_question_id=1, answer="a", citations=[cite()])
    ev_u = Evidence(sub_question_id=2, answer="b", citations=[])
    cr1 = Critique(sub_question_id=1, overall_confidence=0.8)
    cr2 = Critique(sub_question_id=2, overall_confidence=0.4)
    # grounded_share=0.5, mean_conf=0.6 -> 0.5*0.5 + 0.5*0.6 = 0.55
    assert Synthesizer.groundedness([ev_g, ev_u], [cr1, cr2]) == 0.55
    assert Synthesizer.groundedness([], []) == 0.0


def test_groundedness_penalizes_unsupported_claims():
    from researchpilot.schemas import ClaimVerdict
    ev = Evidence(sub_question_id=1, answer="a", citations=[cite()])
    cr = Critique(sub_question_id=1, overall_confidence=1.0,
                  verdicts=[ClaimVerdict(claim="x", verdict="unsupported")] * 4)
    assert Synthesizer.groundedness([ev], [cr]) == 0.8  # 1.0 - 4*0.05


def test_synthesizer_prompt_includes_flags():
    captured = {}

    class SpyLLM(FakeLLM):
        def generate_text(self, prompt, temperature=0.4):
            captured["prompt"] = prompt
            return "final"

    from researchpilot.schemas import ClaimVerdict
    llm = SpyLLM()
    out = Synthesizer(llm).synthesize(
        "Main?", [SubQuestion(id=1, text="Q1?")],
        [Evidence(sub_question_id=1, answer="ans", citations=[cite()])],
        [Critique(sub_question_id=1, overall_confidence=0.9,
                  verdicts=[ClaimVerdict(claim="shaky", verdict="uncertain",
                                         reason="thin evidence")])])
    assert out == "final"
    assert "shaky" in captured["prompt"] and "thin evidence" in captured["prompt"]


# ----------------------------------------------- SDK response parsing (llm)
def _grounded_resp():
    web = SimpleNamespace(uri="https://x.org", title="X")
    chunk = SimpleNamespace(web=web)
    gm = SimpleNamespace(grounding_chunks=[chunk], web_search_queries=["q1", "q2"])
    cand = SimpleNamespace(grounding_metadata=gm)
    return SimpleNamespace(text="hello", candidates=[cand])


def test_parse_grounded_response():
    text, cites, queries = GeminiClient.parse_grounded_response(_grounded_resp())
    assert text == "hello"
    assert cites[0].uri == "https://x.org" and cites[0].title == "X"
    assert queries == ["q1", "q2"]


def test_parse_grounded_response_no_metadata():
    resp = SimpleNamespace(text="hi", candidates=[SimpleNamespace(grounding_metadata=None)])
    text, cites, queries = GeminiClient.parse_grounded_response(resp)
    assert text == "hi" and cites == [] and queries == []


def test_parse_function_calls():
    fc = SimpleNamespace(name="flag_claim", args={"claim": "c", "verdict": "unsupported"})
    part = SimpleNamespace(function_call=fc)
    resp = SimpleNamespace(candidates=[SimpleNamespace(
        content=SimpleNamespace(parts=[part, SimpleNamespace(function_call=None)]))])
    calls = GeminiClient.parse_function_calls(resp)
    assert calls == [("flag_claim", {"claim": "c", "verdict": "unsupported"})]


def test_extract_json_tolerates_fences_and_prose():
    assert GeminiClient._extract_json('[1, 2]') == [1, 2]
    assert GeminiClient._extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert GeminiClient._extract_json('Sure! Here it is: [{"text": "q"}]') == [{"text": "q"}]
