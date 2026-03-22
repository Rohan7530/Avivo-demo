"""Prompt builder: assembles the final LLM prompt for the RAG pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_PROMPT = (
    "You are AvivaBot, a helpful and accurate enterprise AI assistant. "
    "Answer questions using ONLY the provided context. "
    "If the context does not contain the answer, say: "
    "'I don't have enough information in my knowledge base to answer that.' "
    "Always cite the source document name at the end of your answer as: "
    "Source: [document name]."
)


def build_prompt(
    query: str,
    context_chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]] | None = None,
) -> str:
    """
    Build the full prompt string for the LLM.

    Args:
        query: The user's current question.
        context_chunks: Retrieved chunks from ChromaDB, each with 'text' and 'source'.
        history: Optional list of prior turns, each dict with 'role' and 'content'.

    Returns:
        A fully formatted prompt string ready to send to Ollama.
    """
    history = history or []

    prompt_parts: List[str] = [
        f"SYSTEM:\n{SYSTEM_PROMPT}\n",
    ]

    # History section (last 3 turns = up to 6 messages)
    if history:
        prompt_parts.append("CONVERSATION HISTORY:")
        for turn in history[-6:]:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt_parts.append("")

    # Context section
    if context_chunks:
        prompt_parts.append("RELEVANT CONTEXT FROM KNOWLEDGE BASE:")
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "unknown")
            text = chunk.get("text", "")
            prompt_parts.append(f"[{i}] (Source: {source})\n{text}")
        prompt_parts.append("")

    # User query
    prompt_parts.append(f"USER QUESTION:\n{query}\n")
    prompt_parts.append("ASSISTANT ANSWER:")

    return "\n".join(prompt_parts)


def extract_source_attribution(context_chunks: List[Dict[str, Any]]) -> str:
    """
    Build a deduplicated source attribution string from context chunks.

    Args:
        context_chunks: Retrieved chunks with 'source' metadata.

    Returns:
        Formatted source string like "policy.md, tech_faqs.md".
    """
    sources = list(dict.fromkeys(c.get("source", "unknown") for c in context_chunks))
    return ", ".join(sources)
