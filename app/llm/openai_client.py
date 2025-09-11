import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tools.registry import FUNCTION_MAP, TOOLS_JSON

# Initialize OpenAI client
client = OpenAI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
available_models = ["gpt-4o-mini", "gpt-5", "gpt-4o", "o3", "o4"]


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    used_tools: List[Tuple[str, Any]]


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
    prompt,
    sys_prompt=None,
    tools=TOOLS_JSON,
    function_map=FUNCTION_MAP,
    model=DEFAULT_MODEL,
    temperature=0.1,
    max_tokens=500,
) -> LLMResponse:
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

        # R1: let the model decide if it wants tools
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

        # If no tools were called, we’re done
        if not tool_calls:
            usage = getattr(r1, "usage", None)
            return LLMResponse(
                text=m1.content or "",
                model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                completion_tokens=(
                    getattr(usage, "completion_tokens", 0) if usage else 0
                ),
                total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
                used_tools=[],
            )

        # Execute tool calls
        tool_messages = []
        used_tools = []
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

        # R2: feed tool results back
        r2 = client.chat.completions.create(
            model=model,
            messages=messages
            + [
                {
                    "role": "assistant",
                    "content": m1.content or "",
                    "tool_calls": m1.tool_calls,
                }
            ]
            + tool_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        m2 = r2.choices[0].message
        usage2 = getattr(r2, "usage", None)

        return LLMResponse(
            text=m2.content or "",
            model=model,
            prompt_tokens=getattr(usage2, "prompt_tokens", 0) if usage2 else 0,
            completion_tokens=getattr(usage2, "completion_tokens", 0) if usage2 else 0,
            total_tokens=getattr(usage2, "total_tokens", 0) if usage2 else 0,
            used_tools=used_tools,
        )

    except Exception as e:
        return LLMResponse(
            text=f"Error calling LLM: {e}",
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )


if __name__ == "__main__":
    print("🤖 OpenAI Client Test")
    print("=" * 50)

    # Mock search result for testing
    mock_results = [
        {
            "title": "Heart Disease Overview",
            "text": "Heart disease is a leading cause of death. Symptoms of heart disease can include chest pain or pressure, shortness of breath, pain in the jaw, neck, back, or arms, fatigue, dizziness, lightheadedness, and swelling in the feet or ankles. Heart disease can also cause palpitations, nausea, and cold sweats. These symptoms can signal a heart attack, arrhythmia, or heart failure, and any of them should prompt an immediate medical appointment.",
            "source_type": "wikipedia",
            "score": 0.95,
        }
    ]

    test_query = "What are heart attack symptoms?"

    context = (
        "You are a medical llm assistant answering questions to the best of your ability using pub med research papers"
        " and text book documents/shards as your source of information. You job is to bridge the gap between the users"
        " request and the specific highly technical academic speech patterns of your source documentation."
    )
    query_w_data = (
        f"Query:\n{test_query} \n Top result from hybrid search: {mock_results}"
    )
    print(query_w_data)
    print(f"\nPrompt:\n{context}")
    print("=" * 50)

    result = llm(prompt=query_w_data, sys_prompt=context)

    print("— Response —")
    print(result.text)
    print("\n— Usage —")
    print(f"Model: {result.model}")
    print(
        f"Prompt tokens: {result.prompt_tokens} | Completion tokens: {result.completion_tokens} | Total: {result.total_tokens}"
    )
