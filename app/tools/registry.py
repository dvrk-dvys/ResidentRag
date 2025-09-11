from app.search.hybrid_search import hybrid_search
from tools.search_wikipedia import WikipediaTool
from tools.search_pubmed import PubMedTool
from typing import Dict, Any


WIKI = WikipediaTool()
PUBMED = PubMedTool()

# app/tools/simple_tools.py


def simple_response_ok(query: str) -> Dict[str, Any]:
    """
    Signal to the LLM that this is general/non-medical chat
    and it should answer directly without retrieval.

    Returning a small JSON payload is clearer to the model
    than just the string "OK".
    """
    return {
        "status": "ok",
        "action": "answer_directly",
        "note": "No retrieval needed.",
        "echo_query": query,
    }


def tool_hybrid_search(self, query, top_k=5):
    try:
        hits = hybrid_search(query, top_k=top_k)
    except Exception as e:
        print(f"Error in hybrid search: {e}")
        return []
    return [{"id": h.id,
             "title": h.title,
             "text": h.text,
             "rrf_score": getattr(h, "rrf_score", 0.0),
             "source_type": getattr(h, "source_type", None)} for h in hits]




FUNCTION_MAP = {
    "hybrid_search": WIKI.tool_hybrid_search,
    "wikipedia_search": tool_hybrid_search,
    "pubmed_search": PUBMED.pubmed_semantic_search,
    "simple_response_ok": lambda query: "OK",
}


# app/tools/openai_tools.py
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
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
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
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
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
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
       "type": "function",
       "function": {
         "name": "simple_response_ok",
         "description": "Signal this is general chat; answer directly without retrieval.",
         "parameters": {
           "type":"object",
           "properties":{"query":{"type":"string"}},
           "required":["query"],
           "additionalProperties": False
         }
       }
     }
]
