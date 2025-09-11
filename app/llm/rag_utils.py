import os
import json
from llm.openai_client import llm, client
from search.hybrid_search import hybrid_search, Hit
from typing import List, Dict


#todo CHAT HISTORY/MEMORY

#Use only the facts from the DATA. If the context doesn't contain sufficient information, say so.

#<QUESTION>
#{question}
#</QUESTION>

#<DATA>
#{data}
#</DATA>
#
#


medical_context_template = """

You are a medical llm assistant answering questions to the best of your ability using pub med research papers and text book
documents/shards as your source of information. Your job is to bridge the gap between the users request and the specific
highly technical academic speech patterns of your source documentation.

{user_context} {detail_context}
""".strip()
#????? {conversation_history}

old_basic_llm_prompt_template = """
Question: {question}

Please provide a helpful conversational chat response based on your training knowledge. Use your expertise to give accurate, evidence-based information
while being appropriately cautious about medical advice. This is for general chat, pleasantries, clarifications, 
or discussions about previous/future medical queries - not for direct medical questions that require literature search.
""".strip()

basic_llm_prompt_template = """
Answer the user’s question briefly and directly.

Question: {question}
""".strip()


search_prompt_template = """
Question: {question}

Context from medical literature:
{context_data}

Please answer the question based on the provided context and data sources from our ElasticSearch + Qdrant hybrid retriever.
If the data doesn't contain sufficient information, acknowledge this and provide your best approximation of the answer using the information that is available.
""".strip()

#prompt = medical_agent_prompt.format(
#    question=question,
#    context=context,
#    search_queries="\n".join(search_queries),
#    previous_actions='\n'.join([json.dumps(a) for a in previous_actions]),
#    iteration_number=iteration,
#    max_iterations=max_iterations
#)

OLD_medical_agent_prompt = """
You are a medical research assistant. Your goal is to answer: {question}

AVAILABLE ACTIONS:
- MEDICAL_SEARCH: Search medical literature/knowledge base
- CONVERSATION_SEARCH: Search previous conversation context  
- FINAL_ANSWER: Provide complete answer with sources
- CLARIFICATION_NEEDED: Ask user for more specific information

CURRENT CONTEXT:
{context}

PREVIOUS SEARCHES: {search_queries}
PREVIOUS ACTIONS: {previous_actions}

ITERATION: {iteration_number}/{max_iterations}

Respond with JSON in this format:
{{
    "reasoning": "Why you chose this action",
    "action": "MEDICAL_SEARCH|CONVERSATION_SEARCH|FINAL_ANSWER|CLARIFICATION_NEEDED",
    "keywords": ["search", "terms"] // if MEDICAL_SEARCH
    "context_query": "search terms" // if CONVERSATION_SEARCH  
    "answer": "detailed response with citations" // if FINAL_ANSWER
    "clarification_request": "what you need to know" // if CLARIFICATION_NEEDED
    "confidence": 0.8 // how confident you are (0-1)
}}
"""


# === System prompt (tool-aware, compact) ===
MEDICAL_AGENT_SYSTEM = """
ROLE:
You are a careful medical research assistant.

TASK:
- Answer user questions accurately and succinctly.
- Prefer medical primary/secondary sources and cite them when available.

CONTEXT:
{user_context} {detail_context}

TOOLS (use only if needed):
- hybrid_search(query, top_k=5): Retrieve medical literature from ElasticSearch+Qdrant (preferred for clinical/mechanistic queries).
- wikipedia_lookup(query, top_k=5): Retrieve short background/definitions when medical coverage is thin or user needs lay context.
- pubmed_lookup(query, top_k=5): Pull recent/targeted abstracts from PubMed when recency or specificity is important.
- simple_response_ok(query): Signal that this is general chat; answer directly without retrieval.

REASONING (internal to you; don’t expose unless asked):
- First decide if you can answer directly from general knowledge.
- If not, call exactly one tool that best fits the intent. Avoid chaining tools unless strictly necessary.
- If tool output is low-relevance, say so briefly and answer with the best available information.

OUTPUT FORMAT:
- If you used a tool, synthesize a brief answer (3–6 sentences) and list compact citations with titles (and URLs if provided).
- If you did not use a tool, answer briefly and clearly, and avoid invented citations.
- Use plain language for patients; use technical phrasing for clinicians/researchers.

STOP CONDITIONS:
- Stop once you’ve answered the question directly and, if applicable, provided ≤3 high-value citations.
"""

# === User/message prompt (injects conversation + optional retrieved context) ===
MEDICAL_AGENT_USER = """
QUESTION:
{question}

CURRENT CONTEXT:
{context}

PREVIOUS SEARCHES:
{search_queries}

PREVIOUS ACTIONS:
{previous_actions}

INSTRUCTIONS:
- Use tools only if they materially improve the answer.
- If you call a tool, do so once with minimal arguments (usually the user’s question).
- After any tool result is returned, write the final answer that cites those results.
"""




def build_settings_context(settings: Dict) -> Dict[str, str]:
    """Convert UI settings into prompt context strings"""

    user_contexts = {
        "Healthcare Provider": "Provide clinical-level information suitable for medical professionals.",
        "Medical Researcher": "Focus on research findings, methodologies, and scientific evidence.",
        "Patient": "Explain in simple terms that a patient can understand. Always recommend consulting a healthcare provider."
    }

    detail_contexts = {
        "Simple": "Keep responses concise and easy to understand.",
        "Detailed": "Provide comprehensive explanations with relevant details.",
        "Technical": "Include technical details, medical terminology, and specific mechanisms."
    }

    return {
        "user_context": user_contexts.get(settings["user_type"]),
        "detail_context": detail_contexts.get(settings["response_detail"]),
        "show_sources": settings["show_sources"]
    }


def build_rag_context(settings: Dict) -> str:
    """Build context string with source formatting based on user preferences"""
    settings_context = build_settings_context(settings=settings)
    return medical_context_template.format(
        user_context=settings_context["user_context"],
        detail_context=settings_context["detail_context"]
    )


def build_rag_prompt(question: str, settings: Dict, search_results: List[Hit] = None) -> str:
    """Build RAG prompt with search results using template injection"""
    
    if not search_results:
        context_text = "No medical literature search was performed for this query."
        return basic_llm_prompt_template.format(
            question=question,
            context_data=context_text
        )

    elif settings["show_sources"]:
        context_text = "\n\n".join([
            f"Source: {doc.title}\nContent: {doc.text}\nRelevance Score: {doc.rrf_score:.3f}"
            for doc in search_results
        ])
    else:
        context_text = "\n\n".join([doc.text for doc in search_results])

    return search_prompt_template.format(
        question=question,
        context_data=context_text
    )






#--------------------------------------------------------------------------------

def build_prompt(query, search_results):
    """
    Build a medical RAG prompt from query and search results
    """
    prompt_template = """
You're a medical assistant. Answer the QUESTION based on the CONTEXT from medical literature. Use only the facts from the CONTEXT when answering the QUESTION. 

If you cannot find a direct answer, try to formulate a response based on what is medically relevant in the CONTEXT. Always be cautious with medical advice and suggest consulting healthcare professionals when appropriate.

If the CONTEXT completely doesn't contain the answer or anything similar, output NONE.

QUESTION: {question}

CONTEXT: {context}
""".strip()

    context = ""

    for doc in search_results:
        context += f"Source: {doc.get('source_type', 'unknown')}\n"
        context += f"Title: {doc.get('title', '')}\n"
        context += f"Content: {doc.get('text', '')}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt



def rag_with_search(query, search_function, **search_kwargs):
    """
    Generic RAG function that works with any search method
    """
    # Get search results
    search_results = search_function(query, **search_kwargs)

    # Build prompt
    prompt = build_prompt(query, search_results)

    # Get LLM answer
    answer = llm(prompt)

    return {
        'query': query,
        'answer': answer,
        'search_results': search_results,
        'num_results': len(search_results)
    }




# Build context from search results
def build_medical_context(search_results: List[Hit]) -> str:
    context = ""
    for doc in search_results:
        context += f"source: {doc.source_type}\ntitle: {doc.title}\ncontent: {doc.text}\n\n"
    return context.strip()

# Phase 1: Simple Medical RAG
medical_prompt_template = """
You're a medical AI assistant. Answer the QUESTION based on the CONTEXT from medical literature.
Use only the facts from the CONTEXT. If the context doesn't contain sufficient information, say so.

<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()

def simple_medical_rag(question: str) -> str:
    search_results = hybrid_search(question, top_k=5)
    #search_results = None
    context = build_medical_context(search_results)
    
    messages = [{"role": "user", "content": medical_prompt_template.format(
        question=question,
        context=context
    )}]
    
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages
    )
    
    return response.choices[0].message.content

# Phase 2: Decision Medical RAG
medical_decision_template = """
You're a medical AI assistant.

<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>

If CONTEXT is EMPTY or insufficient for a medical question, search for more information:
{{
"action": "SEARCH",
"reasoning": "<why you need more medical information>"
}}

If CONTEXT contains sufficient medical information:
{{
"action": "ANSWER", 
"answer": "<your evidence-based answer>",
"sources": ["pubmed", "textbook", "wikipedia"]
}}
""".strip()

def decision_medical_rag(question: str) -> dict:
    search_results = hybrid_search(question, top_k=5)
    #search_results = None

    context = build_medical_context(search_results)
    
    messages = [{"role": "user", "content": medical_decision_template.format(
        question=question,
        context=context
    )}]
    
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"action": "ANSWER", "answer": response.choices[0].message.content}

# Phase 3: Agentic Medical RAG
medical_agentic_template = """
You're a medical AI assistant specializing in evidence-based responses.

<QUESTION>
{question}
</QUESTION>

<PREVIOUS_SEARCHES>
{search_queries}
</PREVIOUS_SEARCHES>

<CONTEXT>
{context}
</CONTEXT>

<PREVIOUS_ACTIONS>
{previous_actions}
</PREVIOUS_ACTIONS>

Current iteration: {iteration_number}/{max_iterations}

Actions available:
- SEARCH: Find more medical literature  
- ANSWER_EVIDENCE: Answer using provided medical evidence
- ANSWER_INSUFFICIENT: Acknowledge insufficient evidence

{{
"action": "SEARCH",
"reasoning": "<medical reasoning>", 
"keywords": ["medical term 1", "drug name", "condition"]
}}
""".strip()

def agentic_medical_rag(question: str, max_iterations: int = 3) -> dict:
    search_queries = []
    previous_actions = []
    context = ""
    
    for iteration in range(1, max_iterations + 1):
        messages = [{"role": "user", "content": medical_agentic_template.format(
            question=question,
            search_queries='\n'.join(search_queries),
            context=context,
            previous_actions='\n'.join(previous_actions),
            iteration_number=iteration,
            max_iterations=max_iterations
        )}]
        
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
        except:
            result = {"action": "ANSWER_EVIDENCE", "answer": response.choices[0].message.content}
        
        if result.get("action") == "SEARCH":
            query = result.get("keywords", [question])[0] if result.get("keywords") else question
            search_results = hybrid_search(query, top_k=5)
            #search_results = None

            context += build_medical_context(search_results) + "\n\n"
            search_queries.append(query)
            previous_actions.append(f"SEARCH: {query}")
        else:
            return result
    
    return {"action": "ANSWER_INSUFFICIENT", "answer": "Could not find sufficient information after multiple searches"}

if __name__ == "__main__":
    while True:
        question = input("Medical question: ")
        if question == 'stop':
            break
            
        print("\n=== Simple RAG ===")
        simple_answer = simple_medical_rag(question)
        print(simple_answer)
        
        print("\n=== Decision RAG ===") 
        decision_result = decision_medical_rag(question)
        print(decision_result)
        
        print("\n=== Agentic RAG ===")
        agentic_result = agentic_medical_rag(question)
        print(agentic_result)
        print("-" * 50)