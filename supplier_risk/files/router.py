# ============================================================
# router.py — Intent Router
# ============================================================
from langgraph.graph import MessagesState
from helpers import extract_text_from_last_message


def router(state: MessagesState) -> str:
    text = extract_text_from_last_message(state).lower().strip()

    if "status" in text:
        return "status_node"
    if text in ["hi", "hello"]:
        return "greet_node"
    if text == "help":
        return "help_node"
    if "test db" in text:
        return "test_db_node"
    if "sample supplier" in text:
        return "sample_supplier_node"

    # 7 analytics commands
    if "risk profile" in text or ("risk" in text and any(s in text for s in ["acme","vertex","silkroad","blueridge","novatech","coastal","sunrise","apex","prism","delta"])):
        return "supplier_risk_profile_node"
    if "high risk" in text or "risky supplier" in text or "flagged supplier" in text:
        return "high_risk_suppliers_node"
    if "open po" in text or "po exposure" in text or "pipeline" in text or "at risk" in text:
        return "open_po_exposure_node"
    if "delivery" in text or "on-time" in text or "on time" in text or "performance" in text:
        return "delivery_performance_node"
    if "news" in text or "alert" in text or "recent event" in text:
        return "supplier_news_alerts_node"
    if "esg" in text or "compliance" in text or "sustainability" in text or "audit" in text:
        return "esg_compliance_node"
    if "category" in text or "electronics" in text or "logistics" in text or "raw material" in text:
        return "category_risk_briefing_node"

    # Natural language fallback
    question_starters = ["who", "what", "where", "why", "how", "which", "show", "tell", "find", "analyze", "can you", "i want", "i need"]
    if any(text.startswith(q) for q in question_starters) or len(text.split()) >= 4:
        return "ask_node"

    return "help_node"
