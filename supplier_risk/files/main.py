# ============================================================
# main.py — SupplierRiskAgent Entry Point
# ============================================================
import traceback

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from nodes_core import (
    log_user_message, status_node, greet_node, help_node,
    test_db_node, sample_supplier_node,
)
from nodes_analytics import (
    supplier_risk_profile_node, high_risk_suppliers_node,
    open_po_exposure_node, delivery_performance_node,
    supplier_news_alerts_node, esg_compliance_node,
    category_risk_briefing_node,
)
from nodes_nlp import ask_node
from router import router


class AgentBasic:
    def __init__(self) -> None:
        self.graph = None

    def setup(self) -> None:
        g = StateGraph(MessagesState)

        g.add_node("logger",                        log_user_message)
        g.add_node("status_node",                   status_node)
        g.add_node("greet_node",                    greet_node)
        g.add_node("help_node",                     help_node)
        g.add_node("test_db_node",                  test_db_node)
        g.add_node("sample_supplier_node",          sample_supplier_node)
        g.add_node("supplier_risk_profile_node",    supplier_risk_profile_node)
        g.add_node("high_risk_suppliers_node",      high_risk_suppliers_node)
        g.add_node("open_po_exposure_node",         open_po_exposure_node)
        g.add_node("delivery_performance_node",     delivery_performance_node)
        g.add_node("supplier_news_alerts_node",     supplier_news_alerts_node)
        g.add_node("esg_compliance_node",           esg_compliance_node)
        g.add_node("category_risk_briefing_node",   category_risk_briefing_node)
        g.add_node("ask_node",                      ask_node)

        g.add_edge(START, "logger")

        g.add_conditional_edges(
            "logger",
            router,
            {
                "status_node":                  "status_node",
                "greet_node":                   "greet_node",
                "help_node":                    "help_node",
                "test_db_node":                 "test_db_node",
                "sample_supplier_node":         "sample_supplier_node",
                "supplier_risk_profile_node":   "supplier_risk_profile_node",
                "high_risk_suppliers_node":     "high_risk_suppliers_node",
                "open_po_exposure_node":        "open_po_exposure_node",
                "delivery_performance_node":    "delivery_performance_node",
                "supplier_news_alerts_node":    "supplier_news_alerts_node",
                "esg_compliance_node":          "esg_compliance_node",
                "category_risk_briefing_node":  "category_risk_briefing_node",
                "ask_node":                     "ask_node",
            },
        )

        for node in [
            "status_node", "greet_node", "help_node", "test_db_node",
            "sample_supplier_node", "supplier_risk_profile_node",
            "high_risk_suppliers_node", "open_po_exposure_node",
            "delivery_performance_node", "supplier_news_alerts_node",
            "esg_compliance_node", "category_risk_briefing_node", "ask_node",
        ]:
            g.add_edge(node, END)

        self.graph = g.compile()

    async def invoke(self, user_query: str, **kwargs):
        try:
            if self.graph is None:
                self.setup()
            messages = {"messages": [{"role": "user", "content": user_query}]}
            return await self.graph.ainvoke(messages)
        except Exception as e:
            tb = traceback.format_exc()
            print("INVOKE ERROR:", repr(e))
            print(tb)
            return {"messages": [AIMessage(content=f"ERROR: {e}\n\nTRACE:\n{tb}")]}


agent = AgentBasic()