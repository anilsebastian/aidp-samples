# ============================================================
# nodes_core.py — Core nodes (JSON output for UI)
# ============================================================
import json
import traceback
from langgraph.graph import MessagesState

from config import CATALOG_KEY, SCHEMA_KEY, SPEND_TABLE
from helpers import now_utc, format_sqltool_result, extract_rows_from_result
from sql_tools import make_sql_tool, AIDP_AVAILABLE, IMPORT_ERRORS as SQL_IMPORT_ERRORS
from llm import IMPORT_ERRORS as LLM_IMPORT_ERRORS


def json_response(data: dict) -> dict:
    """Wrap response data in standard message format."""
    return {"messages": [{"role": "ai", "content": json.dumps(data)}]}


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
    
    return json_response({
        "answer_type": "status",
        "title": "Agent Status",
        "status": "error" if all_errors else "ok",
        "errors": all_errors if all_errors else [],
        "config": {
            "catalog_key": CATALOG_KEY,
            "schema_key": SCHEMA_KEY,
            "spend_table": SPEND_TABLE
        }
    })


def greet_node(state: MessagesState):
    return json_response({
        "answer_type": "greeting",
        "content": "Hello! I'm your Spend Optimization Agent. Ask me anything about your procurement spend.",
        "suggested_commands": [
            {"label": "Top Suppliers", "command": "top suppliers"},
            {"label": "Spend by Category", "command": "spend by category"},
            {"label": "Maverick Spend", "command": "maverick spend"},
            {"label": "Savings Opportunities", "command": "savings opportunities"}
        ]
    })


def help_node(state: MessagesState):
    return json_response({
        "answer_type": "help",
        "title": "Spend Optimization Agent",
        "description": "Ask me anything about your procurement spend using natural language.",
        "example_questions": [
            "Who are my biggest vendors?",
            "Where am I at risk?",
            "Where can I save money?",
            "What should I consolidate?"
        ],
        "commands": [
            {"name": "top suppliers", "description": "Top 10 suppliers by spend", "category": "analytics"},
            {"name": "maverick spend", "description": "Purchases without contracts or unmatched suppliers", "category": "analytics"},
            {"name": "spend leakage", "description": "Consolidation opportunities across systems", "category": "analytics"},
            {"name": "spend by category", "description": "Breakdown by category with off-contract %", "category": "analytics"},
            {"name": "savings opportunities", "description": "Price variance across business units", "category": "analytics"},
            {"name": "supplier risk", "description": "Concentration risk and single-source categories", "category": "analytics"},
            {"name": "spend trend", "description": "Monthly spend trends over time", "category": "analytics"},
            {"name": "sample spend", "description": "View sample data (5 rows)", "category": "utility"},
            {"name": "test db", "description": "Verify database connectivity", "category": "utility"},
            {"name": "status", "description": "Check agent configuration", "category": "utility"}
        ]
    })


def time_node(state: MessagesState):
    user_text = extract_text_from_last_message(state)
    return json_response({
        "answer_type": "text",
        "content": f"You asked: '{user_text}'. Current time is {now_utc()}"
    })


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
        
        return json_response({
            "answer_type": "status",
            "title": "Database Connectivity Test",
            "status": "ok",
            "message": "Database connection successful",
            "result": format_sqltool_result(result)
        })
    except Exception as e:
        tb = traceback.format_exc()
        print("test_db_node error:", e)
        print(tb)
        return json_response({
            "answer_type": "status",
            "title": "Database Connectivity Test",
            "status": "error",
            "message": str(e),
            "trace": tb
        })


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
        rows = extract_rows_from_result(result)
        
        # Get column names from first row
        columns = list(rows[0].keys()) if rows else []
        
        return json_response({
            "answer_type": "table",
            "title": f"Sample Data from {SPEND_TABLE}",
            "data": rows,
            "columns": columns,
            "row_count": len(rows)
        })
    except Exception as e:
        tb = traceback.format_exc()
        print("sample_spend_node error:", e)
        print(tb)
        return json_response({
            "answer_type": "error",
            "error": str(e),
            "trace": tb
        })


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
        rows = extract_rows_from_result(result)
        
        table_names = [row.get("TABLE_NAME", row.get("table_name", "")) for row in rows]
        
        return json_response({
            "answer_type": "list",
            "title": f"Tables in {CATALOG_KEY}.{SCHEMA_KEY}",
            "items": table_names,
            "count": len(table_names)
        })
    except Exception as e:
        tb = traceback.format_exc()
        print("list_tables_node error:", e)
        print(tb)
        return json_response({
            "answer_type": "error",
            "error": str(e),
            "trace": tb,
            "tip": "Run `test db` to validate connectivity."
        })
