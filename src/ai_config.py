"""Single place that decides which AI provider the whole system uses.

Set AI_PROVIDER in .env to "ollama" (free, local - default) or "openai".
Nothing else in the codebase imports a provider package directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SUPPORTED_PROVIDERS = ("ollama", "openai")
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").strip().lower()
if AI_PROVIDER not in SUPPORTED_PROVIDERS:
    raise ValueError(f"AI_PROVIDER must be one of {SUPPORTED_PROVIDERS}, got '{AI_PROVIDER}'.")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# temperature 0 = most deterministic answers; we want facts, not creativity.
TEMPERATURE = 0.0

# Context window for local models. llama3.2 defaults to 131k tokens, which needs ~18 GB of
# RAM and spills onto the CPU on a 16 GB laptop. Our prompts (system prompt + tool schemas +
# 3 retrieved chunks + short history) stay well under 4k tokens, so 8k is plenty.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))


def _require_openai_key() -> None:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-your"):
        raise RuntimeError("AI_PROVIDER=openai but OPENAI_API_KEY in .env is still a placeholder.")


def get_chat_model():
    """LangChain chat model used by the agent."""
    if AI_PROVIDER == "openai":
        _require_openai_key()
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=TEMPERATURE)

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        num_ctx=OLLAMA_NUM_CTX,
    )


def configure_llamaindex() -> None:
    """Point LlamaIndex's global Settings at the chosen embedding model and LLM."""
    from llama_index.core import Settings

    if AI_PROVIDER == "openai":
        _require_openai_key()
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI

        Settings.embed_model = OpenAIEmbedding(model=OPENAI_EMBED_MODEL)
        Settings.llm = OpenAI(model=OPENAI_CHAT_MODEL, temperature=TEMPERATURE)
        return

    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama

    # nomic-embed-text was trained with task prefixes; using them improves retrieval.
    nomic = "nomic" in OLLAMA_EMBED_MODEL
    Settings.embed_model = OllamaEmbedding(
        model_name=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        query_instruction="search_query: " if nomic else None,
        text_instruction="search_document: " if nomic else None,
    )
    Settings.llm = Ollama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        context_window=OLLAMA_NUM_CTX,
        request_timeout=180.0,
    )


def storage_dir() -> Path:
    """Each provider gets its own index folder: vectors from different embedding
    models are not compatible, so they must never be mixed."""
    return PROJECT_ROOT / "storage" / AI_PROVIDER
