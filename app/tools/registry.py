import inspect
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

from search.hybrid_search import hybrid_search
from search.search_utils import Hit
from tools.search_pubmed import PubMedTool
from tools.search_wikipedia import WikipediaTool

WIKI = WikipediaTool()
PUBMED = PubMedTool()


def simple_response_ok(query):
    """
    Signal to the LLM that this is general/non-medical chat
    and it should answer directly without retrieval.

    Returning a small JSON payload is clearer to the llm
    than just the string "OK".
    """
    return {
        "status": "ok",
        "action": "answer_directly",
        "note": "No retrieval needed.",
        "echo_query": query,
    }


def tool_hybrid_search(query, top_k=5, local=False):
    hits: list[Hit] = hybrid_search(query, top_k=top_k, local=local)
    return [
        {
            "id": h.id,
            "title": h.title,
            "text": h.text,
            "rrf_score": float(getattr(h, "rrf_score", 0.0)),
            "source_type": getattr(h, "source_type", None),
        }
        for h in hits
    ]


FUNCTION_MAP = {
    "hybrid_search": tool_hybrid_search,
    "wikipedia_search": WIKI.wiki_semantic_search,
    "pubmed_search": PUBMED.pubmed_semantic_search,
    # "simple_response_ok": simple_response_ok,
}

TOOLS_JSON = [
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "Search the MedRAG corpus via ES+Qdrant hybrid retriever.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User question"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                    },
                    "local": {
                        "type": "boolean",
                        "description": "Use local URLs instead of Docker service names",
                        "default": False,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Fetch short, relevant Wikipedia passages for definitions/background.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": "Search PubMed and return top abstract chunks for the query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    # {
    #    "type": "function",
    #    "function": {
    #        "name": "simple_response_ok",
    #        "description": "Signal this is general chat; answer directly without retrieval.",
    #        "parameters": {
    #            "type": "object",
    #            "properties": {"query": {"type": "string"}},
    #            "required": ["query"],
    #            "additionalProperties": False,
    #        },
    #    },
    # },
]
