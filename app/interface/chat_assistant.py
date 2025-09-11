from typing import Dict

import markdown
from llm.openai_client import llm
from llm.rag_utils import build_rag_context, build_rag_prompt
from search.hybrid_search import hybrid_search


class ChatAssistant:
    def __init__(
        self, tools, developer_prompt="You are a helpful medical Resident Assistant."
    ):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.chat_messages = [{"role": "developer", "content": self.developer_prompt}]
        self.llm = llm
        self.hybrid_search = hybrid_search

    def query_llm(self, prompt, sys_prompt):
        return self.llm(
            prompt=prompt,
            sys_prompt=sys_prompt,
            # tools=self.tools.get_tools(),
        )

    def process_message(self, question, settings: Dict):
        if question.strip().lower() == "stop":
            #! add a tool to gracefully handle stopping
            return {"answer": "Chat ended."}

        self.chat_messages.append(
            {"role": "user", "content": question}
        )  # todo: add the results of the llm with tools to the messages

        # Build context based on user type and detail level
        sys_prompt = build_rag_context(settings)

        # Get search results
        # search_results = self.hybrid_search(question, top_k=5)
        search_results = None

        # Build RAG prompt
        prompt = build_rag_prompt(question, settings, search_results)  #! todo
        response = self.query_llm(prompt=prompt, sys_prompt=sys_prompt)
        sources = []

        if settings.get("show_sources") and search_results:
            sources = [{"title": d.title, "score": d.rrf_score} for d in search_results]

        resp = {
            "answer": response.text,
            "model": response.model,
            "tokens": response.total_tokens,
        }
        if sources:  # <- only attach when non-empty
            resp["sources"] = sources
        return resp

    def bind(self):
        """maybe use this in the wikipedia api cascade? Connect widgets to this assistant."""

        # self.chat_interface.register_on_submit(self.handle_user_message)
        pass

    def display_function_call(self, entry, result):
        pass

    def display_response(self, entry):
        pass
