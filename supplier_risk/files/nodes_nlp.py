# ============================================================
# nodes_nlp.py — Natural Language Query Node
# ============================================================
import traceback
from langgraph.graph import MessagesState
from helpers import extract_text_from_last_message, format_sqltool_result
from sql_tools import make_sql_tool
from llm import ensure_llm, LLM_AVAILABLE
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import CATALOG_KEY, SCHEMA_KEY, SUPPLIER_MASTER_TABLE, SPEND_HISTORY_TABLE, PO_PIPELINE_TABLE

SCHEMA_DESC = f"""
Three tables in {CATALOG_KEY}.{SCHEMA_KEY}:

1. {SUPPLIER_MASTER_TABLE}: supplier_id, supplier_name, country, category, tier, payment_terms, active_since, risk_score_internal (VARCHAR, cast with TO_NUMBER)
2. {SPEND_HISTORY_TABLE}: supplier_id, quarter, spend_usd, po_count, on_time_delivery_pct, defect_rate_pct, payment_delays (all VARCHAR, cast with TO_NUMBER)
3. {PO_PIPELINE_TABLE}: po_id, supplier_id, item_description, po_value_usd, expected_delivery, criticality, sourcing_alternative

SQL Rules:
- Double-quote all column names: "supplier_name"
- Cast numeric columns: TO_NUMBER("risk_score_internal")
- Use FETCH FIRST N ROWS ONLY (not LIMIT)
- JOIN on "supplier_id" between tables
"""

NLP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"""You are a SQL expert for a supplier risk database.
Schema: {SCHEMA_DESC}
Generate a single valid Oracle SQL SELECT query to answer the user's question.
Return ONLY the SQL query, no explanation, no markdown fences."""),
    ("user", "{question}")
])


async def ask_node(state: MessagesState):
    user_text = extract_text_from_last_message(state)
    try:
        if not LLM_AVAILABLE:
            return {"messages": [{"role": "ai", "content": "LLM not available. Use specific commands like `high risk suppliers` or type `help`."}]}

        llm   = ensure_llm()
        chain = NLP_PROMPT | llm | StrOutputParser()
        sql   = await chain.ainvoke({"question": user_text})
        sql   = sql.strip().replace("```sql", "").replace("```", "").strip()

        tool   = make_sql_tool("ask_query", "NLP query", sql)
        result = await tool.ainvoke({})
        table  = format_sqltool_result(result)

        msg = f"## 🔍 Query Results\n\n**Generated SQL:**\n```sql\n{sql}\n```\n\n{table}"
        return {"messages": [{"role": "ai", "content": msg}]}
    except Exception as e:
        tb = traceback.format_exc()
        return {"messages": [{"role": "ai", "content": f"❌ Error: {e}\n\nTry specific commands like `high risk suppliers`. Type `help` for options.\n\nTrace: {tb}"}]}
