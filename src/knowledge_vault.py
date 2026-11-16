"""Grounded LlamaIndex knowledge vault over knowledge_manual/.

Pipeline: SimpleDirectoryReader -> SentenceSplitter (with overlap) -> embeddings ->
VectorStoreIndex, persisted to ./storage/<provider>/ so it is only rebuilt when the
documents or chunk settings change. The query engine is exposed to LangChain as the
`search_afyaplus_guidelines` @tool.

Grounding guarantees:
  * Retrieved chunks below SIMILARITY_CUTOFF are discarded (no weak matches).
  * If nothing relevant survives, the tool returns NOT_FOUND without calling the LLM,
    so the model has nothing to hallucinate from.
  * Every answer ends with the source file names it was built from.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache

from langchain_core.tools import tool
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor

import ai_config

logger = logging.getLogger("afyaplus.vault")

KNOWLEDGE_DIR = ai_config.PROJECT_ROOT / "knowledge_manual"
NOT_FOUND = "Information not found in AfyaPlus guidelines"

# Token management - see README "Token management" for the reasoning.
# Values were tuned by measuring retrieval scores on sample questions (see README).
CHUNK_SIZE = 256        # tokens per chunk: about one guideline section -> precise matches
CHUNK_OVERLAP = 40      # ~15% overlap so a rule split across a boundary stays readable
TOP_K = 3               # 3 x 256 = ~770 context tokens: small, cheap, fits llama3.2 easily
SIMILARITY_CUTOFF = 0.50  # off-topic questions scored <= 0.46; real ones scored >= 0.58

QA_TEMPLATE = PromptTemplate(
    "You are the AfyaPlus guidelines lookup service.\n"
    "Context from AfyaPlus documents:\n"
    "---------------------\n{context_str}\n---------------------\n"
    "Answer the question using ONLY the context above. Copy numbers, limits and doses "
    "exactly as written. If the context does not contain the answer, reply with exactly: "
    f"{NOT_FOUND}\n"
    "Question: {query_str}\nAnswer: "
)


def _fingerprint() -> str:
    """Hash of document contents + settings. Changes => index must be rebuilt."""
    h = hashlib.sha256()
    for path in sorted(KNOWLEDGE_DIR.glob("*")):
        if path.suffix in {".md", ".txt"}:
            h.update(path.name.encode())
            h.update(path.read_bytes())
    h.update(f"{CHUNK_SIZE}-{CHUNK_OVERLAP}-{Settings.embed_model.model_name}".encode())
    return h.hexdigest()


def build_or_load_index(force_rebuild: bool = False) -> VectorStoreIndex:
    """Load the persisted index, or (re)build it if missing or out of date."""
    ai_config.configure_llamaindex()
    persist_dir = ai_config.storage_dir()
    manifest = persist_dir / "manifest.json"
    fingerprint = _fingerprint()

    if not force_rebuild and manifest.exists():
        if json.loads(manifest.read_text()).get("fingerprint") == fingerprint:
            logger.info("Loading persisted index from %s", persist_dir)
            storage = StorageContext.from_defaults(persist_dir=str(persist_dir))
            return load_index_from_storage(storage)
        logger.info("Knowledge manual changed - rebuilding index.")

    documents = SimpleDirectoryReader(
        input_dir=str(KNOWLEDGE_DIR), required_exts=[".md", ".txt"]
    ).load_data()
    if not documents:
        raise RuntimeError(f"No .md/.txt documents found in {KNOWLEDGE_DIR}")

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info("Indexed %d documents into %d chunks.", len(documents), len(nodes))

    index = VectorStoreIndex(nodes)
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    manifest.write_text(json.dumps({"fingerprint": fingerprint, "chunks": len(nodes)}))
    return index


@lru_cache(maxsize=1)
def get_query_engine():
    """Build the query engine once per process (lazy, so importing is cheap)."""
    index = build_or_load_index()
    return index.as_query_engine(
        similarity_top_k=TOP_K,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)],
        text_qa_template=QA_TEMPLATE,
    )


def query_guidelines(question: str) -> str:
    """Ask the knowledge vault a question; returns a grounded answer with sources."""
    response = get_query_engine().query(question)
    sources = sorted({n.node.metadata.get("file_name", "unknown") for n in response.source_nodes})

    # Strict source-context validation: no relevant chunks => no answer.
    if not response.source_nodes:
        return NOT_FOUND
    answer = str(response).strip()
    if not answer or NOT_FOUND.lower() in answer.lower():
        return NOT_FOUND
    return f"{answer}\nSources: {', '.join(sources)}"


@tool(parse_docstring=True)
def search_afyaplus_guidelines(query: str) -> str:
    """Search the official AfyaPlus guidelines (insurance cover, waiting periods, exclusions,
    pre-authorisation, clinic referral and triage rules, medication dosing values, BMI bands).

    Use this for EVERY policy, coverage, routing or dosing question. Pass a short, specific
    question. The result ends with 'Sources:' listing the files it came from, or says
    'Information not found in AfyaPlus guidelines' if the documents do not cover it.

    Args:
        query: A focused question, e.g. "Bronze plan outpatient limit" or "paracetamol dose per kg".
    """
    try:
        if not query or not query.strip():
            return "Error: query must not be empty."
        return query_guidelines(query.strip())
    except Exception as exc:  # e.g. Ollama not running - report, don't crash the agent
        logger.error("Guideline search failed: %s", type(exc).__name__)  # no message: may echo input
        return f"Error: the guidelines service is unavailable ({type(exc).__name__})."
