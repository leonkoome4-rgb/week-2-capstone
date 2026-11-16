"""LangChain 1.x orchestration core for the AfyaPlus assistant.

create_agent() wires together:
  * the chat model chosen in ai_config (Ollama llama3.2 or OpenAI),
  * three tools: the grounded guidelines search + two maths tools,
  * InMemorySaver checkpointer -> per-patient memory keyed by thread_id,
  * SummarizationMiddleware -> keeps long conversations inside the token budget.

The agent only ever receives MASKED text (see main.py); it never sees real PII.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolCallRequest,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

import ai_config
from knowledge_vault import NOT_FOUND, search_afyaplus_guidelines
from tools import calculate_weight_based_dose

SYSTEM_PROMPT = f"""You are the AfyaPlus Health assistant for insurance verification and clinic routing.

STRICT RULES:
1. For ANY question about insurance cover, limits, waiting periods, exclusions, referrals,
   triage or medication doses, call the search_afyaplus_guidelines tool first.
2. Answer ONLY from tool results. Never invent or guess a dose, limit, price or policy rule.
3. For a medication dose, call calculate_weight_based_dose with the medication name and the
   patient's weight in kg. It looks up the official dosing values itself.
   Never do arithmetic yourself; report the tool's numbers exactly.
4. If the guidelines tool returns "{NOT_FOUND}", reply with exactly that sentence and nothing
   else about the topic.
5. End every guideline-based answer with the "Sources:" line from the tool result.
   If you used no guidelines, do not write a Sources line.
6. Questions about this conversation itself (e.g. "what did I tell you?", "which plan am I
   on?") are answered from the chat history WITHOUT calling any tool. If the patient never
   told you something, say so.
7. Phone numbers and emails appear as codes like [MASKED_PHONE_1] or [MASKED_EMAIL_1].
   Treat each code as the real value itself. Example: patient said "my phone is
   [MASKED_PHONE_1]" and later asks for their number -> answer "Your phone number is
   [MASKED_PHONE_1]." Copy codes exactly. Never comment on the codes or call them placeholders.
8. If the patient describes an emergency red flag, tell them to go to the nearest emergency
   department immediately.
9. Be brief and clear. This is a training system, not a substitute for a clinician."""

# calculate_bmi is kept in tools.py (tested) but not given to the agent: llama3.2 kept
# inventing BMI calls on unrelated turns. One maths tool = no tool confusion.
TOOLS = [search_afyaplus_guidelines, calculate_weight_based_dose]
MAX_AGENT_STEPS = 12          # ~5 tool round-trips per turn is plenty for these questions
MAX_TOOL_CALLS_PER_TURN = 5
MATH_TOOL_NAMES = {calculate_weight_based_dose.name, "calculate_bmi"}


def _number_in_text(value: float, text: str) -> bool:
    """True if `value` (e.g. 18.0) appears as a number in `text` (e.g. "weighs 18 kg")."""
    for match in re.findall(r"\d+(?:[.,]\d+)?", text):
        if abs(float(match.replace(",", ".")) - value) < 1e-9:
            return True
    return False


@wrap_tool_call
def argument_grounding_guard(request: ToolCallRequest, handler):
    """Block maths-tool calls whose numbers were never given by the patient.

    Small models sometimes invent a weight or height "to be helpful". Every numeric argument
    to a maths tool must appear in something the patient actually typed on this thread.
    """
    call = request.tool_call
    if call["name"] in MATH_TOOL_NAMES:
        patient_text = " ".join(
            str(m.content) for m in request.state["messages"] if isinstance(m, HumanMessage)
        )
        for arg, value in call["args"].items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue  # non-numeric args (e.g. medication name) are validated by the tool
            if not _number_in_text(number, patient_text):
                return ToolMessage(
                    content=(
                        f"Error: {arg}={value} was not provided by the patient, so this call "
                        "was blocked. STOP calling tools for this question. Reply to the "
                        "patient now using the conversation so far, or ask them for the value."
                    ),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
    return handler(request)


def build_agent(checkpointer: InMemorySaver | None = None):
    """Create the agent. One checkpointer holds the memory of every patient thread."""
    model = ai_config.get_chat_model()
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
        middleware=[
            argument_grounding_guard,
            # Backstop: at most 5 tool calls per turn; extra calls are refused with an error
            # message and the model must answer with what it has.
            ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS_PER_TURN, exit_behavior="continue"),
            # When a thread passes ~3k tokens, older turns are replaced by a summary and the
            # last 6 messages are kept verbatim. Summaries are built from masked text only.
            SummarizationMiddleware(model=model, trigger=("tokens", 3000), keep=("messages", 6)),
        ],
    )


@dataclass
class AgentTurn:
    answer: str                                        # masked answer from the model
    tool_calls: list[str] = field(default_factory=list)  # audit trail: which tools ran


def run_turn(agent, thread_id: str, masked_text: str) -> AgentTurn:
    """Send one masked user message on a patient's thread and return the masked reply."""
    # recursion_limit caps agent steps per turn (LangGraph's default is ~10,000), so a
    # model stuck retrying a blocked tool cannot loop forever.
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_AGENT_STEPS}
    try:
        result = agent.invoke({"messages": [HumanMessage(masked_text)]}, config=config)
    except GraphRecursionError:
        return AgentTurn(
            answer="Sorry, I couldn't complete that request safely. Please rephrase or contact "
            "the AfyaPlus help desk.",
            tool_calls=[f"stopped after {MAX_AGENT_STEPS} agent steps"],
        )
    except Exception as exc:  # e.g. model server down or timed out
        return AgentTurn(
            answer="Sorry, the assistant is temporarily unavailable. Please try again shortly.",
            tool_calls=[f"error: {type(exc).__name__}"],
        )
    messages = result["messages"]

    # Only look at messages produced in THIS turn (after the last human message).
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    this_turn = messages[last_human + 1 :]

    results = {m.tool_call_id: str(m.content) for m in this_turn if isinstance(m, ToolMessage)}
    calls = [
        f"{c['name']}({c['args']})"
        + (" -> BLOCKED/ERROR" if results.get(c["id"], "").startswith("Error:") else "")
        for m in this_turn
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    ]
    final = next((m for m in reversed(this_turn) if isinstance(m, AIMessage) and m.content), None)
    answer = final.content if final else ""

    # Safety net: if the model ignored a NOT_FOUND tool result and answered anyway,
    # fall back to the grounded refusal.
    guideline_results = [
        m.content
        for m in this_turn
        if isinstance(m, ToolMessage) and m.name == search_afyaplus_guidelines.name
    ]
    if guideline_results and all(r == NOT_FOUND for r in guideline_results) and NOT_FOUND not in answer:
        answer = NOT_FOUND

    # Tidy-up: small models sometimes emit "{}" or an empty/"None" Sources line.
    # A reply that is raw JSON is a tool call the model wrote as text - never show it.
    answer = answer.strip()
    if answer in {"null"} or (answer.startswith(("{", "[")) and answer.endswith(("}", "]"))):
        answer = ""
    answer = re.sub(r"\n*Sources:\s*(None|N/A|\[\])?\s*$", "", answer, flags=re.I)
    answer = re.sub(r"Sources:\s*\[([^\]]+)\]", r"Sources: \1", answer)

    # Deterministic citations: if a tool cited sources but the model dropped the line,
    # add it back so every grounded answer is traceable to its documents.
    cited = sorted({
        name.strip()
        for m in this_turn
        if isinstance(m, ToolMessage)
        for line in re.findall(r"Sources:\s*(.+)", str(m.content))
        for name in line.split(",")
    })
    if answer and cited and answer != NOT_FOUND and "Sources:" not in answer:
        answer += f"\n\nSources: {', '.join(cited)}"

    fallback = "Sorry, I couldn't answer that. Could you rephrase your question?"
    return AgentTurn(answer=answer or fallback, tool_calls=calls)
