import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from llm.rag_utils import build_rag_context, build_rag_prompt
from openai import OpenAI
from tools.registry import FUNCTION_MAP, TOOLS_JSON

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_openai_client():
    """Get OpenAI client - lazy instantiation to avoid import-time failures"""
    return OpenAI()


DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
available_models = ["gpt-4o-mini", "gpt-5", "gpt-4o", "o3", "o4"]


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    used_tools: Optional[List[Tuple[str, Any]]] = None


"""

to do, give chat history for context
messages = [
  {"role": "system", "content": "You’re a careful medical assistant. Explain clearly."},
  {"role": "user", "content": "What are early MI symptoms?"}
]

"""


def llm(prompt, sys_prompt=None, model=DEFAULT_MODEL) -> LLMResponse:
    """
    Call OpenAI LLM with the prompt

    "system" – instructions/behavior setup.
    "user" – the user’s prompt.
    "assistant" – previous assistant replies (for context).
    "tool" – the output returned by a tool/function you called (if you use tools).
    """

    try:
        messages = []

        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.1, max_tokens=500
        )

        msg = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tok = getattr(usage, "total_tokens", prompt_tok + completion_tok)

        return LLMResponse(
            text=msg,
            model=model,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            total_tokens=total_tok,
        )

    except Exception as e:
        return LLMResponse(
            text=f"Error calling LLM: {e}",
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )


def agentic_llm(
    query,
    settings,
    tools=TOOLS_JSON,
    function_map=FUNCTION_MAP,
    model=DEFAULT_MODEL,
    temperature=0.1,
    max_tokens=500,
    max_iterations=3,
) -> LLMResponse:
    """
    Call OpenAI LLM with the prompt using an iterative agent loop.

    Process:
    - The model is given a system prompt (behavior setup) and a user prompt (built with context).
    - The loop runs up to `max_iterations` (default: 3).
        • Iteration 0: the model may answer directly or decide to call a tool (ideally HYBRID_SEARCH first).
        • Iteration 1: if more information is needed, the model may call one supplemental tool
          (e.g., PUBMED_SEARCH or WIKIPEDIA_SEARCH).
        • Iteration 2 (final): the model must provide an answer using the accumulated context.
    - Tool calls are executed locally, and their outputs are collected into:
        • `search_results` – evidence gathered from tools
        • `search_queries` – queries issued to tools
        • `previous_actions` – record of which tools were used
    - These collections are passed back into the prompt each iteration so the model can refine its answer.

    Roles:
    - "system" – behavior and task setup
    - "user" – the constructed question + context
    - "assistant" – prior model responses (including tool calls)
    - "tool" – structured outputs from executed tools

    The loop stops early if the model answers without calling a tool, or after the final iteration.
    """

    # Build context based on user type and detail level
    sys_prompt = build_rag_context(settings)

    # Build RAG prompt
    search_results = []
    search_queries = []
    previous_actions = []
    used_tools = []

    try:
        for i in range(max_iterations):

            prompt = build_rag_prompt(
                question=query,
                settings=settings,
                search_results=search_results,
                search_queries=search_queries,
                previous_actions=previous_actions,
                max_iterations=max_iterations,
                iteration_number=i,
            )

            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": prompt})

            # R1: let the model decide if it wants tools
            client = get_openai_client()
            r1 = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            m1 = r1.choices[0].message
            tool_calls = getattr(m1, "tool_calls", None) or []

            # If last try or no tools were called, we’re done
            if i == max_iterations - 1 or not tool_calls:
                usage = getattr(r1, "usage", None)
                return LLMResponse(
                    text=m1.content or "",
                    model=model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    completion_tokens=(
                        getattr(usage, "completion_tokens", 0) if usage else 0
                    ),
                    total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                    used_tools=used_tools,
                )

            # Execute tool calls
            tool_messages = []
            for call in tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                result = function_map[name](**args)
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
                used_tools.append((name, result))
                search_queries.append(str(args))
                previous_actions.append(f"TOOL:{name}({args})")
                if isinstance(result, list):
                    search_results.extend(result)
                elif isinstance(result, dict) and "hits" in result:
                    search_results.extend(result["hits"])

            # feed tool results back
            messages.append(
                {
                    "role": "assistant",
                    "content": m1.content or "",
                    "tool_calls": m1.tool_calls,
                }
            )
            messages.extend(tool_messages)
    except Exception as e:
        return LLMResponse(
            text=f"Error calling LLM: {e}",
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            used_tools=None,
        )


if __name__ == "__main__":
    # os.environ["ES_URL"] = "http://localhost:9200"
    # os.environ["QDRANT_URL"] = "http://localhost:6333"

    # from elasticsearch import Elasticsearch
    # from qdrant_client import QdrantClient
    # from search.es_search import wait_for_es

    print("ES_URL:", os.getenv("ES_URL"))
    print("QDRANT_URL:", os.getenv("QDRANT_URL"))

    print("🤖 OpenAI Client Test")
    print("=" * 50)

    def in_docker():
        return os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "1"

    # ES_URL = os.getenv("ES_URL") or ("http://elasticsearch:9200" if in_docker() else "http://localhost:9200")
    # ES_CLIENT = Elasticsearch(ES_URL, request_timeout=30)
    # wait_for_es(ES_CLIENT, timeout=30)

    # QDRANT_URL = os.getenv("QDRANT_URL") or ("http://qdrant:6333" if in_docker() else "http://localhost:6333")
    # QDRANT = QdrantClient(url=QDRANT_URL, timeout=30.0)

    # Settings like your UI
    settings = {
        "user_type": "Healthcare Provider",  # "Healthcare Provider" | "Medical Researcher" | "Patient"
        "response_detail": "Detailed",  # "Simple" | "Detailed" | "Technical"
        "show_sources": True,
    }

    test_query = "What are the connections between prolonged corticosteroid use and femoral avascular necrosis?"

    # Run the agent loop (max_iterations defaults to 3 inside agentic_llm)
    result = agentic_llm(
        query=test_query,
        settings=settings,
        model=DEFAULT_MODEL,
        temperature=0.1,
        max_tokens=500,
    )

    print("— Response —")
    print(result.text)

    print("\n— Usage —")
    print(f"Model: {result.model}")
    print(
        f"Prompt tokens: {result.prompt_tokens} | "
        f"Completion tokens: {result.completion_tokens} | "
        f"Total: {result.total_tokens}"
    )

    if result.used_tools:
        print("\n— Tools used —")
        for name, _payload in result.used_tools:
            print(f"- {name}")
    else:
        print("\n— Tools used — none")
