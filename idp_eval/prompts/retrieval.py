"""Prompt and strict schema for one-call retrieval relevance judging."""

from __future__ import annotations


_RETRIEVAL_RELEVANCE_SYSTEM_V1 = """\
Evaluate every ranked retrieved document independently against the same query.

A document is relevant only when it contains information that would materially
help answer, resolve, or correctly support the query. A document is irrelevant
when it is unrelated, has only superficial keyword/entity overlap, does not
materially help answer the query, or is misleading or contradictory to what the
query needs.

For every supplied document return:
- its rank exactly as supplied;
- relevant: a boolean semantic judgment; and
- a concise reason for that judgment.

Preserve every rank exactly. Return exactly one judgment for every supplied
document. Do not add, remove, merge, reorder, or rewrite documents. Do not return
numeric relevance scores, aggregate retrieval metrics, percentages, confidence,
DCG, IDCG, nDCG, reciprocal rank, or hit rate. Python computes all metrics."""


RETRIEVAL_RELEVANCE_SCHEMA_V1 = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer"},
                    "relevant": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["rank", "relevant", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["documents"],
    "additionalProperties": False,
}


def render_retrieval_relevance_prompt(
    query: str, document_texts: list[str]
) -> list[dict[str, str]]:
    """Renders query and ranked document text without retrieval metadata."""
    ranked_documents = "\n\n".join(
        f"[RANK {rank}]\n{text}"
        for rank, text in enumerate(document_texts, start=1)
    )
    user = (
        "[BEGIN DATA]\n\n"
        f"[QUERY]\n{query}\n\n"
        f"[RANKED DOCUMENTS]\n{ranked_documents}\n\n"
        "[END DATA]"
    )
    return [
        {"role": "system", "content": _RETRIEVAL_RELEVANCE_SYSTEM_V1},
        {"role": "user", "content": user},
    ]
