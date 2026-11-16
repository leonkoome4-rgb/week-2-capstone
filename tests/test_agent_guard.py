"""Tests for the agent's argument-grounding guard (no AI model needed)."""

from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import _number_in_text, argument_grounding_guard


def _request(name: str, args: dict, patient_text: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "call_1", "type": "tool_call"},
        tool=None,
        state={"messages": [HumanMessage(patient_text), AIMessage("ok")]},
        runtime=None,
    )


def _handler(request):
    return ToolMessage(content="RAN", tool_call_id=request.tool_call["id"])


def test_number_in_text():
    assert _number_in_text(18.0, "my child weighs 18 kg")
    assert _number_in_text(72.5, "weight 72.5kg")
    assert not _number_in_text(55.0, "my phone is [MASKED_PHONE_1]")


def test_guard_allows_numbers_the_patient_gave():
    req = _request("calculate_weight_based_dose", {"medication": "paracetamol", "weight_kg": "18"},
                   "She weighs 18 kg")
    assert argument_grounding_guard.wrap_tool_call(req, _handler).content == "RAN"


def test_guard_blocks_invented_numbers():
    req = _request("calculate_bmi", {"weight_kg": 55, "height_cm": 150}, "Which plan am I on?")
    result = argument_grounding_guard.wrap_tool_call(req, _handler)
    assert result.content.startswith("Error:")
    assert "not provided by the patient" in result.content


def test_guard_ignores_non_math_tools():
    req = _request("search_afyaplus_guidelines", {"query": "Bronze plan 30000"}, "hello")
    assert argument_grounding_guard.wrap_tool_call(req, _handler).content == "RAN"
