# router.py
# LLM Router — classifies user intent and coordinates agent calls
#
# Satisfies:
#   FR1 - NLP: accepts and understands natural language queries
#   FR4 - Query Routing: classifies intent and delegates to the correct agent
#   FR5 - Conversational Ability: passes conversation history for context-aware responses
#   NFR1 - Hallucination Resistance: LLM only formats retrieved results — it cannot
#           introduce product information not already present in the agent output
#   NFR3 - Modularity: router is the only file that imports both agents

import ollama

from analyst_agent import format_analyst_results, run_analyst_agent
from config import LLM_MODEL, LLM_SYSTEM_PROMPT, MEMORY_WINDOW
from rag_agent import format_rag_results, run_rag_agent

# ── Classification prompt ──────────────────────────────────────────────────────
# WHY a dedicated classification call: a single combined prompt ("classify AND answer")
# is harder to control — the LLM can drift between tasks. Two focused calls
# (classify → format) are more reliable and independently testable (NFR3).

_CLASSIFY_SYSTEM = (
    "You are a query classifier for an e-commerce assistant. "
    "Reply with exactly one word: 'search', 'analytical', or 'chitchat'. Nothing else."
)

_CLASSIFY_USER = """\
Classify the user's latest query:

Reply 'chitchat' if it is:
- A greeting (hello, hi, hey)
- Small talk or thanks
- A general question not about products
- Asking what the assistant can do

Reply 'analytical' if the PRIMARY goal is computing a statistic or aggregate:
- Average/mean price of a category ("average price of watches")
- Counting items ("how many footwear products are there?")
- Ranking ALL items by price ("cheapest 5 watches", "most expensive clothing")
- Brand comparisons by quantity ("which brands have the most products?")
- NOTE: "cheapest N <category>" with no other description = analytical

Reply 'search' if the PRIMARY goal is finding/discovering specific products:
- Product search with a budget: "watches under 700", "shoes under 500", "gifts under 2000"
- Description or feature based: "comfortable running shoes", "blue formal shirt"
- Occasion or intent: "gift for my dad", "something for a wedding", "gym wear"
- Recommendations: "suggest", "what's good for", "I need something for"
- KEY RULE: if the query is "[product type] under/below/above [price]" with no explicit
  statistical operation (average, count, rank all) → it is ALWAYS 'search'

Recent conversation (for context on follow-up queries):
{history_text}

Latest query: "{query}"

One word:"""

# ── Response formatting prompt ─────────────────────────────────────────────────
# WHY temperature 0.3 here (not 0): classification must be deterministic (temp=0),
# but the final response benefits from slight variation in phrasing to feel
# conversational rather than robotic (FR5). Still low enough to stay grounded.

_FORMAT_USER = """\
The user asked: "{query}"

Here are the results:
{results}

Write a natural, conversational response as a helpful shopping assistant. Rules:
- Never mention "database", "criteria", "data", "system", or "records"
- Never apologise or say nothing was found if products ARE listed above
- Do NOT say the user had "criteria" — just answer naturally
- If fewer results than requested are shown, say "here's what I found" — no excuses
- NEVER invent product names, brands, or prices — only use what is listed above
- Keep it short: one friendly intro sentence, the list, one closing offer"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """
    Gradio 6 stores message content as a list of typed blocks, e.g.:
        [{"text": "hello", "type": "text"}]
    Ollama expects a plain string. This normalises either format.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    return str(content)


def _history_to_messages(history: list[dict]) -> list[dict]:
    """
    Trim history to MEMORY_WINDOW messages and normalise content to plain strings
    so Ollama's Pydantic validator accepts them.
    """
    return [
        {"role": msg["role"], "content": _extract_text(msg.get("content", ""))}
        for msg in history[-MEMORY_WINDOW:]
    ]


def _history_to_text(history: list[dict]) -> str:
    """Compact text summary of recent turns for the classification prompt."""
    if not history:
        return "(none)"
    lines = []
    for msg in history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {_extract_text(msg.get('content', ''))[:120]}")
    return "\n".join(lines)


# ── Classification ─────────────────────────────────────────────────────────────

def _classify(query: str, history: list[dict]) -> str:
    """
    Returns 'search' or 'analytical'.
    History is included so follow-up queries ("show me cheaper ones") route correctly.
    """
    prompt = _CLASSIFY_USER.format(
        history_text=_history_to_text(history),
        query=query,
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _CLASSIFY_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        think=False,
        options={"temperature": 0},
    )

    label = response.message.content.strip().lower()
    if "analytical" in label:
        return "analytical"
    if "chitchat" in label:
        return "chitchat"
    return "search"


# ── Main router ────────────────────────────────────────────────────────────────

def route(query: str, history: list[dict] | None = None) -> tuple[str, str]:
    """
    Full routing pipeline: classify → agent → format.

    Args:
        query:   Current user message
        history: Gradio conversation history (list of role/content dicts)

    Returns:
        Tuple of (response_text, intent) where intent is 'search' or 'analytical'.
        The intent is returned so app.py can display a source label in the UI.
    """
    if history is None:
        history = []

    # ── Step 1: Classify ───────────────────────────────────────────────────────
    intent = _classify(query, history)

    # ── Chitchat: bypass agents, respond conversationally ─────────────────────
    if intent == "chitchat":
        messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
        messages += _history_to_messages(history)
        messages.append({"role": "user", "content": query})
        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            think=False,
        options={"temperature": 0.5},
        )
        text = response.message.content.strip().replace("Kart AI", "Kart")
        return text, intent

    # ── Step 2: Run agent ──────────────────────────────────────────────────────
    if intent == "analytical":
        result      = run_analyst_agent(query)
        results_text = format_analyst_results(result)
    else:
        products     = run_rag_agent(query)
        results_text = format_rag_results(products)
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    messages += _history_to_messages(history)
    messages.append({
        "role": "user",
        "content": _FORMAT_USER.format(query=query, results=results_text),
    })

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        think=False,
        options={"temperature": 0.3},
    )

    return response.message.content.strip(), intent


# ── Streaming variant ──────────────────────────────────────────────────────────

def route_stream(query: str, history: list[dict] | None = None):
    """
    Generator version of route(). Classifies and runs the agent synchronously,
    then streams the final LLM formatting response chunk by chunk.

    Yields:
        (text_chunk: str, intent: str) tuples.
        intent is determined once before streaming begins.
    """
    if history is None:
        history = []

    intent = _classify(query, history)

    # Chitchat: skip agents entirely, respond conversationally
    if intent == "chitchat":
        messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
        messages += _history_to_messages(history)
        messages.append({"role": "user", "content": query})
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            think=False,
        options={"temperature": 0.5},
            stream=True,
        )
        for chunk in stream:
            yield (chunk.message.content or "").replace("Kart AI", "Kart"), intent
        return

    if intent == "analytical":
        result       = run_analyst_agent(query)
        results_text = format_analyst_results(result)
    else:
        products     = run_rag_agent(query)
        results_text = format_rag_results(products)

    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    messages += _history_to_messages(history)
    messages.append({
        "role": "user",
        "content": _FORMAT_USER.format(query=query, results=results_text),
    })

    stream = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        think=False,
        options={"temperature": 0.3},
        stream=True,
    )

    for chunk in stream:
        yield chunk.message.content or "", intent


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        # (query, simulated_history)
        ("Find me a good pair of sports shoes",        []),
        ("What is the average price of watches?",      []),
        ("Show me the 5 cheapest items in Jewellery",  []),
        # Follow-up: should classify as 'search' in context of previous shoe query
        ("Do you have anything waterproof?",           [
            {"role": "user",      "content": "Find me a good pair of sports shoes"},
            {"role": "assistant", "content": "Here are some sports shoes..."},
        ]),
    ]

    for query, history in test_cases:
        print(f"\nQuery   : {query}")
        print(f"History : {len(history)} messages")
        response, intent = route(query, history)
        print(f"Intent  : {intent}")
        print(f"Response: {response}")
        print("-" * 60)
# Alias for app.py import
route_query = route