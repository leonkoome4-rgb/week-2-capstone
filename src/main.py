"""AfyaPlus end-to-end pipeline:  raw input -> mask -> agent -> unmask -> patient.

Run the scripted demo:      python src/main.py
Chat interactively:         python src/main.py --chat
Force an index rebuild:     python src/main.py --rebuild
"""

from __future__ import annotations

import logging
import sys
import warnings

from masking import PIIMasker, PIIRedactingFilter


def setup_logging() -> None:
    """Every log line passes through the PII filter - masking at every exit."""
    handler = logging.StreamHandler()
    handler.addFilter(PIIRedactingFilter())
    handler.setFormatter(logging.Formatter("  [log] %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    for noisy in ("httpx", "httpcore", "llama_index", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")


class AfyaPlusPipeline:
    """Holds one agent (shared memory store) and one PII vault per patient thread."""

    def __init__(self) -> None:
        from agent import build_agent

        self.agent = build_agent()
        self.vaults: dict[str, PIIMasker] = {}

    def ask(self, patient_id: str, raw_text: str, verbose: bool = True) -> str:
        from agent import run_turn

        masker = self.vaults.setdefault(patient_id, PIIMasker())

        masked_input = masker.mask(raw_text)                      # 1. privacy middleware
        turn = run_turn(self.agent, patient_id, masked_input)     # 2. agent (masked only)
        final_answer = masker.unmask(turn.answer)                 # 3. compliance de-masking

        if verbose:
            print(f"\n  WHAT THE AI SAW : {masked_input}")
            print(f"  TOOLS CALLED    : {', '.join(turn.tool_calls) or 'none'}")
            print(f"  AI REPLY (masked): {turn.answer}")
            print(f"\n  ASSISTANT >> {final_answer}")
        return final_answer


DEMO = [
    ("patient-001", "Turn 1 - patient shares PII and asks a cover question",
     "Hello, I'm Wanjiru. My phone is 0712 345 678 and my email is wanjiru.kamau@example.co.ke. "
     "I'm on the AfyaPlus Bronze plan. What is my annual outpatient limit and co-payment?"),
    ("patient-001", "Turn 2 - memory check (nothing repeated by the patient)",
     "Which plan did I say I'm on, and what phone number should the clinic call me on?"),
    ("patient-001", "Turn 3 - RAG + maths tool",
     "My daughter weighs 18 kg and has a fever. According to the AfyaPlus dosing guidelines, "
     "what paracetamol dose in mg and syrup volume in mL should she get?"),
    ("patient-001", "Turn 4 - question the documents cannot answer",
     "What are the AfyaPlus office opening hours in Kisumu?"),
    ("patient-002", "Different patient - memory is isolated per thread",
     "What phone number did I give you earlier?"),
]


def run_demo(pipeline: AfyaPlusPipeline) -> None:
    for patient_id, title, text in DEMO:
        print("\n" + "=" * 90)
        print(f"[{patient_id}] {title}")
        print(f"  PATIENT >> {text}")
        pipeline.ask(patient_id, text)
    print("\n" + "=" * 90)


def run_chat(pipeline: AfyaPlusPipeline) -> None:
    patient_id = input("Patient ID (e.g. patient-001): ").strip() or "patient-001"
    print("Type your question (or 'quit').")
    while True:
        text = input("\nYOU >> ").strip()
        if text.lower() in {"quit", "exit", ""}:
            break
        pipeline.ask(patient_id, text)


def main() -> None:
    setup_logging()
    import ai_config
    from knowledge_vault import build_or_load_index, get_query_engine

    print(f"AfyaPlus assistant | provider: {ai_config.AI_PROVIDER}")
    if "--rebuild" in sys.argv:
        build_or_load_index(force_rebuild=True)
    get_query_engine()  # load (or build) the vault once, up front

    pipeline = AfyaPlusPipeline()
    if "--chat" in sys.argv:
        run_chat(pipeline)
    else:
        run_demo(pipeline)


if __name__ == "__main__":
    main()
