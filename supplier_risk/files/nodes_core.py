# ============================================================
# nodes_core.py — Utility Nodes
# ============================================================
import traceback
from langgraph.graph import MessagesState
from config import CATALOG_KEY, SCHEMA_KEY, SUPPLIER_MASTER_TABLE, SPEND_HISTORY_TABLE, PO_PIPELINE_TABLE
from helpers import extract_text_from_last_message, format_sqltool_result
from sql_tools import make_sql_tool, AIDP_AVAILABLE, IMPORT_ERRORS as SQL_IMPORT_ERRORS
from llm import IMPORT_ERRORS as LLM_IMPORT_ERRORS


def log_user_message(state: MessagesState):
    print("User asked:", extract_text_from_last_message(state))
    return state


def status_node(state: MessagesState):
    all_errors = SQL_IMPORT_ERRORS + LLM_IMPORT_ERRORS
    if all_errors:
        msg = "⚠️ IMPORT ERRORS:\n" + "\n".join(f"- {x}" for x in all_errors)
    else:
        msg = "✅ Status OK: all imports succeeded."
    msg += (
        f"\n\n**Catalog:** {CATALOG_KEY}"
        f"\n**Schema:** {SCHEMA_KEY}"
        f"\n**Tables:** {SUPPLIER_MASTER_TABLE}, {SPEND_HISTORY_TABLE}, {PO_PIPELINE_TABLE}"
        f"\n**Knowledge Base:** supplier_risk_kb"
    )
    return {"messages": [{"role": "ai", "content": msg}]}


def greet_node(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "👋 Welcome to the **Supplier Risk Intelligence Agent**. I can help you monitor and assess risk across your supplier base. Type `help` to see all commands."}]}


def help_node(state: MessagesState):
    msg = (
        "## 🏭 Supplier Risk Intelligence Agent\n\n"
        "**Analytics Commands:**\n"
        "- `supplier risk profile [name]` — Full risk profile: scores, trends, open POs, news\n"
        "- `high risk suppliers` — All suppliers above risk threshold\n"
        "- `open po exposure` — Open PO pipeline ranked by supplier risk\n"
        "- `delivery performance [name]` — Quarterly delivery trend for a supplier\n"
        "- `supplier news alerts [name]` — Latest risk alerts and news\n"
        "- `esg compliance status` — ESG and compliance status across all suppliers\n"
        "- `category risk briefing [category]` — Risk briefing for Electronics, Logistics, etc.\n\n"
        "**Utility Commands:**\n"
        "- `test db` — Test database connectivity\n"
        "- `sample supplier` — Show sample supplier data\n"
        "- `status` — Show agent configuration\n\n"
        "**Examples:**\n"
        "- `supplier risk profile acme`\n"
        "- `delivery performance silkroad`\n"
        "- `category risk briefing electronics`"
    )
    return {"messages": [{"role": "ai", "content": msg}]}


async def test_db_node(state: MessagesState):
    try:
        tool   = make_sql_tool("test_db", "Test DB", "SELECT 1 AS test_val FROM DUAL")
        result = await tool.ainvoke({})
        return {"messages": [{"role": "ai", "content": "✅ Database connection successful.\n\n" + format_sqltool_result(result)}]}
    except Exception as e:
        tb = traceback.format_exc()
        return {"messages": [{"role": "ai", "content": f"❌ DB Test Failed: {e}\n\n{tb}"}]}


async def sample_supplier_node(state: MessagesState):
    try:
        query  = f'SELECT "supplier_id", "supplier_name", "country", "category", TO_NUMBER("risk_score_internal") AS risk_score FROM {SUPPLIER_MASTER_TABLE} FETCH FIRST 5 ROWS ONLY'
        tool   = make_sql_tool("sample_supplier", "Sample supplier rows", query)
        result = await tool.ainvoke({})
        return {"messages": [{"role": "ai", "content": f"**Sample from {SUPPLIER_MASTER_TABLE}:**\n\n" + format_sqltool_result(result)}]}
    except Exception as e:
        tb = traceback.format_exc()
        return {"messages": [{"role": "ai", "content": f"❌ Sample query failed: {e}\n\n{tb}"}]}
