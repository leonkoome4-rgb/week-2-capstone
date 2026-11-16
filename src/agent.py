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
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import ai_config
from knowledge_vault import NOT_FOUND, search_afyaplus_guidelines
from tools import MATH_TOOLS

SYSTEM_PROMPT = f"""You are the AfyaPlus Health assistant for insurance verification and clinic routing.

STRICT RULES:
1. For ANY question about insurance cover, limits, waiting periods, exclusions, referrals,
   triage or medication doses, call the search_afyaplus_guidelines tool first.
2. Answer ONLY from tool results. Never invent or guess a dose, limit, price or policy rule.
3. For a medication dose, call calculate_weight_based_dose with the medication name and the
   patient's weight in kg. It looks up the official dosing values itself. For BMI, call
   calculate_bmi. Never do arithmetic yourself; report the tool's numbers exactly.
4. If the guidelines tool returns "{NOT_FOUND}", reply with exactly that sentence and nothing
   else about the topic.
5. End every guideline-based answer with the "Sources:" line from the tool result.
   If you used no guidelines, do not write a Sources line.
6. Questions about this conversation itself (e.g. "what did I tell you?", "which plan am I
   on?") are answered from the chat history WITHOUT calling any tool. If the patient never
   told you something, say so.
7. Text like [MASKED_PHONE_1] or [MASKED_EMAIL_1] is a privacy placeholder for the patient's
   real contact details. Repeat a placeholder exactly as written when you need to refer to it.
   Never try to guess or reconstruct the real value.
8. If the patient describes an emergency red flag, tell them to go to the nearest emergency
   department immediately.
9. Be brief and clear. This is a training system, not a substitute for a clinician."""

TOOLS = [search_afyaplus_guidelines, *MATH_TOOLS]


def build_agent(checkpointer: InMemorySaver | None = None):
    """Create the agent. One checkpointer holds the memory of every patient thread."""
    model = ai_config.get_chat_model()
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
        middleware=[
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
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [HumanMessage(masked_text)]}, config=config)
    messages = result["messages"]

    # Only look at messages produced in THIS turn (after the last human message).
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    this_turn = messages[last_human + 1 :]

    calls = [
        f"{c['name']}({c['args']})"
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

    # Tidy-up: drop an empty trailing "Sources:" line (no citation = no Sources line).
    answer = re.sub(r"\n*Sources:\s*$", "", answer.strip())

    return AgentTurn(answer=answer or "Sorry, I could not produce an answer.", tool_calls=calls)
