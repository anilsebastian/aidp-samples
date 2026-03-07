# ============================================================
# nodes_core.py — Core nodes (help, status, greet, time, test_db)
# ============================================================
import traceback
from langgraph.graph import MessagesState

from config import CATALOG_KEY, SCHEMA_KEY, SPEND_TABLE
from helpers import now_utc, format_sqltool_result
from sql_tools import make_sql_tool, AIDP_AVAILABLE, IMPORT_ERRORS as SQL_IMPORT_ERRORS
from llm import IMPORT_ERRORS as LLM_IMPORT_ERRORS


def extract_text_from_last_message(state: MessagesState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "content"):
        return str(last.content)
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(last)


def log_user_message(state: MessagesState):
    print("User asked:", extract_text_from_last_message(state))
    return state


def status_node(state: MessagesState):
    all_errors = SQL_IMPORT_ERRORS + LLM_IMPORT_ERRORS
    if all_errors:
        msg = "IMPORT ERRORS:\n" + "\n".join([f"- {x}" for x in all_errors])
    else:
        msg = "Status OK: imports succeeded."
    msg += (
        f"\n\nCATALOG_KEY={CATALOG_KEY}"
        f"\nSCHEMA_KEY={SCHEMA_KEY}"
        f"\nSPEND_TABLE={SPEND_TABLE}"
    )
    return {"messages": [{"role": "ai", "content": msg}]}


def greet_node(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "Hello! I'm your Spend Optimization Agent. Ask me anything about your procurement spend, or type `help` for options."}]}


def help_node(state: MessagesState):
    msg = (
        "**Spend Optimization Agent**\n\n"
        "💬 **Ask me anything!** Try natural language questions like:\n"
        "- *\"Who are my biggest vendors?\"*\n"
        "- *\"Where am I at risk?\"*\n"
        "- *\"Where can I save money?\"*\n"
        "- *\"What should I consolidate?\"*\n\n"
        "📊 **Analytics Commands:**\n"
        "- `top suppliers` — Top 10 suppliers by spend\n"
        "- `maverick spend` — Flag purchases without contracts or unmatched suppliers\n"
        "- `spend leakage` — Find consolidation opportunities across systems\n"
        "- `spend by category` — Breakdown by category with off-contract %\n"
        "- `savings opportunities` — Price variance across business units\n"
        "- `supplier risk` — Concentration risk and single-source categories\n"
        "- `spend trend` — Monthly spend trends over time\n\n"
        "🔧 **Utility Commands:**\n"
        "- `sample spend` — View sample data (5 rows)\n"
        "- `test db` — Verify database connectivity\n"
        "- `status` — Check agent configuration"
    )
    return {"messages": [{"role": "ai", "content": msg}]}


def time_node(state: MessagesState):
    user_text = extract_text_from_last_message(state)
    return {"messages": [{"role": "ai", "content": f"You asked: '{user_text}'. Current time is {now_utc()}"}]}


async def test_db_node(state: MessagesState):
    """Test DB connectivity with SELECT 1 FROM DUAL."""
    try:
        sql = "SELECT 1 AS test_val FROM DUAL"
        print(f"DEBUG executing SQL: {sql}")
        tool = make_sql_tool(
            name="test_db",
            description="Test database connectivity",
            query=sql,
        )
        result = await tool.ainvoke({})
        return {"messages": [{"role": "ai", "content": "✅ DB Test Successful:\n" + format_sqltool_result(result)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("test_db_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"❌ DB Test Failed: {e}\n\nTRACE:\n{tb}"}]}


async def sample_spend_node(state: MessagesState):
    """Select 5 rows from the spend table."""
    try:
        sql = f"SELECT * FROM {SPEND_TABLE} FETCH FIRST 5 ROWS ONLY"
        print(f"DEBUG executing SQL: {sql}")
        tool = make_sql_tool(
            name="sample_spend",
            description="Sample spend rows",
            query=sql,
        )
        result = await tool.ainvoke({})
        return {"messages": [{"role": "ai", "content": f"Sample from {SPEND_TABLE}:\n" + format_sqltool_result(result)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("sample_spend_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"Sample query failed: {e}\n\nTRACE:\n{tb}"}]}


async def list_tables_node(state: MessagesState):
    """List tables in the schema using Oracle syntax."""
    try:
        sql = "SELECT table_name FROM user_tables ORDER BY table_name FETCH FIRST 50 ROWS ONLY"
        print(f"DEBUG executing SQL: {sql}")
        tool = make_sql_tool(
            name="list_tables",
            description="List tables in schema",
            query=sql,
        )
        result = await tool.ainvoke({})
        return {"messages": [{"role": "ai", "content": f"Tables in {CATALOG_KEY}.{SCHEMA_KEY}:\n" + format_sqltool_result(result)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("list_tables_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"List tables failed: {e}\n\nTRACE:\n{tb}\n\nTip: run `test db` to validate connectivity."}]}
