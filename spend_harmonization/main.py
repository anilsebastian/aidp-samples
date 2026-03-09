# ============================================================
# main.py — AIDP Spend Optimization Agent Entry Point
# ============================================================
import traceback

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

# Import nodes
from nodes_core import (
    log_user_message,
    status_node,
    greet_node,
    help_node,
    time_node,
    test_db_node,
    sample_spend_node,
    list_tables_node,
)
from nodes_analytics import (
    top_suppliers_node,
    maverick_spend_node,
    spend_leakage_node,
    spend_by_category_node,
    savings_opportunities_node,
    supplier_risk_node,
    spend_trend_node,
)
from nodes_nlp import ask_node
from router import router


# ============================================================
# Agent
# ============================================================
class AgentBasic:
    def __init__(self) -> None:
        self.graph = None

    def setup(self) -> None:
        g = StateGraph(MessagesState)

        # Add nodes
        g.add_node("logger", log_user_message)
        g.add_node("status_node", status_node)
        g.add_node("greet_node", greet_node)
        g.add_node("help_node", help_node)
        g.add_node("time_node", time_node)
        g.add_node("test_db_node", test_db_node)
        g.add_node("sample_spend_node", sample_spend_node)
        g.add_node("top_suppliers_node", top_suppliers_node)
        g.add_node("list_tables_node", list_tables_node)
        g.add_node("maverick_spend_node", maverick_spend_node)
        g.add_node("spend_leakage_node", spend_leakage_node)
        g.add_node("spend_by_category_node", spend_by_category_node)
        g.add_node("savings_opportunities_node", savings_opportunities_node)
        g.add_node("supplier_risk_node", supplier_risk_node)
        g.add_node("spend_trend_node", spend_trend_node)
        g.add_node("ask_node", ask_node)

        # Entry point
        g.add_edge(START, "logger")

        # Routing
        g.add_conditional_edges(
            "logger",
            router,
            {
                "status_node": "status_node",
                "greet_node": "greet_node",
                "help_node": "help_node",
                "time_node": "time_node",
                "test_db_node": "test_db_node",
                "sample_spend_node": "sample_spend_node",
                "top_suppliers_node": "top_suppliers_node",
                "list_tables_node": "list_tables_node",
                "maverick_spend_node": "maverick_spend_node",
                "spend_leakage_node": "spend_leakage_node",
                "spend_by_category_node": "spend_by_category_node",
                "savings_opportunities_node": "savings_opportunities_node",
                "supplier_risk_node": "supplier_risk_node",
                "spend_trend_node": "spend_trend_node",
                "ask_node": "ask_node",
            },
        )

        # End edges
        g.add_edge("status_node", END)
        g.add_edge("greet_node", END)
        g.add_edge("help_node", END)
        g.add_edge("time_node", END)
        g.add_edge("test_db_node", END)
        g.add_edge("sample_spend_node", END)
        g.add_edge("top_suppliers_node", END)
        g.add_edge("list_tables_node", END)
        g.add_edge("maverick_spend_node", END)
        g.add_edge("spend_leakage_node", END)
        g.add_edge("spend_by_category_node", END)
        g.add_edge("savings_opportunities_node", END)
        g.add_edge("supplier_risk_node", END)
        g.add_edge("spend_trend_node", END)
        g.add_edge("ask_node", END)

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


# AIDP looks for a variable named `agent`
agent = AgentBasic()
