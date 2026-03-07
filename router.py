# ============================================================
# router.py — Router logic
# ============================================================
from langgraph.graph import MessagesState

from nodes_core import extract_text_from_last_message


def router(state: MessagesState) -> str:
    """Route user message to appropriate node."""
    text = extract_text_from_last_message(state).lower().strip()

    # Exact command matches first
    if "status" in text:
        return "status_node"
    if text in ["hi", "hello"]:
        return "greet_node"
    if text == "help":
        return "help_node"
    if "time" in text:
        return "time_node"
    if "test db" in text:
        return "test_db_node"
    if "sample spend" in text or "probe spend" in text:
        return "sample_spend_node"
    if "top suppliers" in text:
        return "top_suppliers_node"
    if "list tables" in text or "list silver" in text:
        return "list_tables_node"
    if "maverick" in text:
        return "maverick_spend_node"
    if "leakage" in text or "consolidation" in text:
        return "spend_leakage_node"
    if "by category" in text or "category spend" in text:
        return "spend_by_category_node"
    if "saving" in text or "price variance" in text:
        return "savings_opportunities_node"
    if "supplier risk" in text or "concentration" in text or "single source" in text:
        return "supplier_risk_node"
    if "trend" in text or "over time" in text or "monthly" in text:
        return "spend_trend_node"
    
    # Natural language questions - route to LLM intent understanding
    question_starters = ["who", "what", "where", "why", "how", "which", "show me", "tell me", "find", "analyze", "can you", "i want", "i need"]
    if any(text.startswith(q) for q in question_starters) or len(text.split()) >= 4:
        return "ask_node"

    return "help_node"
