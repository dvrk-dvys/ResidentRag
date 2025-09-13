from typing import Dict

import markdown
from llm.openai_client import agentic_llm, llm


class ChatAssistant:
    def __init__(
        self, developer_prompt="You are a helpful medical Resident Assistant."
    ):
        self.developer_prompt = developer_prompt
        self.chat_messages = [{"role": "developer", "content": self.developer_prompt}]

    def query_llm(self, query, settings):
        result, out_citations = agentic_llm(
            query=query,
            settings=settings,
        )

    def process_message(self, question, settings):
        if question.strip().lower() == "stop":
            #! add a tool to gracefully handle stopping
            return {"answer": "Chat ended."}

        self.chat_messages.append(
            {"role": "user", "content": question}
        )  # todo: add the results of the llm with tools to the messages

        response, out_citations = self.query_llm(query=question, settings=settings)

        resp = {
            "answer": response.text,
            "model": response.model,
            "tokens": response.total_tokens,
            "citations": out_citations,
        }
        return resp

    def display_function_call(self, entry, result):
        pass

    def display_response(self, entry):
        pass
