import os
import traceback
import json
import re
from datetime import datetime
from typing import Any, List

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState, START, END

# ============================================================
# Configuration
# ============================================================
CATALOG_KEY = os.getenv("CATALOG_KEY", "custom_fdibundletest")
SCHEMA_KEY = os.getenv("SCHEMA_KEY", "fdi_aidp_cust01")
SPEND_TABLE = os.getenv("SPEND_TABLE", "HARMONIZED_SPEND")

# LLM Configuration (OCI GenAI)
REGION = os.getenv("REGION", "us-phoenix-1")
MODEL_ID = os.getenv("MODEL_ID", "xai.grok-3-mini")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "generic")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..aaaaaaaaoyio62q3gtybcicjloxahujztqz5tn4dgzsrtxybx2smbdv4vhva")

# ============================================================
# Optional AIDP SQLTool imports (guarded)
# ============================================================
IMPORT_ERRORS: List[str] = []
AIDP_AVAILABLE = True

try:
    from aidputils.agents.toolkit.configs import AIDPToolConf
    from aidputils.agents.toolkit.tool_helper import create_langgraph_tool
except Exception as e:
    AIDP_AVAILABLE = False
    IMPORT_ERRORS.append(f"aidputils import failed: {e!r}")

# LLM imports
LLM_AVAILABLE = True
try:
    from aidputils.agents.toolkit.agent_helper import init_oci_llm, pre_invoke_setup
    from aidputils.agents.toolkit.configs import OCIAIConf
except Exception as e:
    LLM_AVAILABLE = False
    IMPORT_ERRORS.append(f"LLM imports failed: {e!r}")

# Module-level LLM instance (lazy init)
_llm_instance = None

def _ensure_llm():
    """Lazy-init the LLM instance."""
    global _llm_instance
    if not LLM_AVAILABLE:
        raise RuntimeError("LLM not available. Check imports.")
    pre_invoke_setup()
    if _llm_instance is None:
        oci_conf = OCIAIConf(
            model_provider=MODEL_PROVIDER,
            compartment_id=OCI_COMPARTMENT_ID,
            model_id=MODEL_ID,
            endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
            model_args={
                "temperature": 0.2,
                "max_tokens": 1024,
            }
        )
        _llm_instance = init_oci_llm(oci_conf)
        print("LLM initialized")
    return _llm_instance

# ============================================================
# Helpers
# ============================================================
def _now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_currency(val) -> str:
    """Format number as currency with $ and commas."""
    if val is None:
        return "-"
    try:
        num = float(val)
        if num >= 1_000_000:
            return f"${num/1_000_000:,.1f}M"
        elif num >= 1_000:
            return f"${num:,.0f}"
        else:
            return f"${num:,.2f}"
    except (ValueError, TypeError):
        return str(val)


def _format_number(val) -> str:
    """Format number with commas."""
    if val is None:
        return "-"
    try:
        num = float(val)
        if num >= 1_000_000:
            return f"{num/1_000_000:,.1f}M"
        elif num >= 1_000:
            return f"{num:,.0f}"
        else:
            return str(int(num)) if num == int(num) else f"{num:.1f}"
    except (ValueError, TypeError):
        return str(val)


def _make_bar(value: float, max_value: float, width: int = 15, color: str = "blue") -> str:
    """Create a text-based bar chart using colored emoji squares."""
    if max_value == 0:
        return ""
    ratio = min(value / max_value, 1.0)
    filled = int(ratio * width)
    empty = width - filled
    
    # Color options using emoji squares
    colors = {
        "blue": "🟦",
        "green": "🟩",
        "orange": "🟧",
        "red": "🟥",
        "purple": "🟪",
        "yellow": "🟨",
    }
    filled_char = colors.get(color, "🟦")
    empty_char = "⬜"
    
    return filled_char * filled + empty_char * empty


def _risk_emoji(risk_type: str) -> str:
    """Return emoji for risk type."""
    if risk_type is None:
        return "⚪"
    risk_lower = str(risk_type).lower()
    if "no contract" in risk_lower and "unmatched" in risk_lower:
        return "🔴"  # High risk
    elif "no contract" in risk_lower:
        return "🟠"  # Medium-high risk
    elif "unmatched" in risk_lower:
        return "🟠"  # Medium-high risk
    elif "low confidence" in risk_lower:
        return "🟡"  # Medium risk
    else:
        return "⚪"


def _clean_column_name(col: str) -> str:
    """Make column names more readable."""
    replacements = {
        "SUPPLIER_NAME": "Supplier",
        "CANONICAL_SUPPLIER_NAME": "Supplier",
        "SUPPLIER": "Supplier",
        "SPEND_USD": "Spend",
        "TOTAL_SPEND": "Total Spend",
        "LINE_AMOUNT_USD": "Amount",
        "CATEGORY_NAME": "Category",
        "BUSINESS_UNIT": "Business Unit",
        "CONTRACT_REFERENCE": "Contract",
        "MATCH_METHOD": "Match Type",
        "MATCH_CONFIDENCE": "Confidence",
        "AVG_CONFIDENCE": "Avg Confidence",
        "RISK_TYPE": "Risk",
        "NUM_SYSTEMS": "# Systems",
        "SYSTEMS": "Systems",
        "TRANSACTION_COUNT": "Transactions",
        "PO_ID": "PO #",
        "PCT_OFF_CONTRACT": "% Off-Contract",
        "UNIQUE_SUPPLIERS": "Suppliers",
    }
    return replacements.get(col.upper(), col.replace("_", " ").title())


def _extract_rows_from_result(result: Any) -> List[dict]:
    """Extract rows from SQLTool result."""
    if result is None:
        return []
    if isinstance(result, dict):
        payload = result
        if isinstance(payload.get("result"), dict):
            payload = payload["result"]
        if isinstance(payload.get("structuredContent"), dict):
            payload = payload["structuredContent"]
        rows = payload.get("rows", [])
        if isinstance(rows, list):
            return rows
    return []


def _extract_text_from_last_message(state: MessagesState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "content"):
        return str(last.content)
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(last)


def _format_sqltool_result(result: Any, as_table: bool = True) -> str:
    """Best-effort formatting for SQLTool outputs — returns markdown table when possible."""
    print(f"DEBUG result type: {type(result)}")
    if isinstance(result, dict):
        print(f"DEBUG result keys: {result.keys()}")
    
    if result is None:
        return "(empty result)"

    if isinstance(result, str):
        return result[:4000]

    if isinstance(result, dict):
        payload = result
        if isinstance(payload.get("result"), dict):
            payload = payload["result"]
        if isinstance(payload.get("structuredContent"), dict):
            payload = payload["structuredContent"]

        rows = payload.get("rows")
        cols = payload.get("columns")

        if rows is None:
            return str(result)[:1500]

        if not rows:
            return "(No rows returned)"

        # rows as list[dict] — format as markdown table
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            if as_table and rows:
                # Get column names from first row
                columns = list(rows[0].keys())
                
                # Build markdown table
                lines = []
                # Header
                lines.append("| " + " | ".join(columns) + " |")
                # Separator
                lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
                # Rows
                for row in rows[:50]:
                    values = []
                    for col in columns:
                        val = row.get(col, "")
                        # Format numbers nicely
                        if isinstance(val, (int, float)):
                            if val >= 1000:
                                val = f"{val:,.0f}"
                            else:
                                val = str(val)
                        else:
                            val = str(val) if val is not None else ""
                        values.append(val)
                    lines.append("| " + " | ".join(values) + " |")
                
                if len(rows) > 50:
                    lines.append(f"\n*... showing 50 of {len(rows)} rows*")
                return "\n".join(lines)
            else:
                lines = [str(r) for r in rows[:50]]
                if len(rows) > 50:
                    lines.append(f"... showing 50 of {len(rows)} rows")
                return "\n".join(lines)

        # rows as list[list]
        col_names: List[str] = []
        if isinstance(cols, list):
            for c in cols:
                if isinstance(c, dict) and "name" in c:
                    col_names.append(str(c["name"]))
                else:
                    col_names.append(str(c))

        if as_table and col_names and rows:
            lines = []
            # Header
            lines.append("| " + " | ".join(col_names) + " |")
            # Separator
            lines.append("| " + " | ".join(["---"] * len(col_names)) + " |")
            # Rows
            for r in rows[:50]:
                values = []
                for val in r:
                    if isinstance(val, (int, float)):
                        if val >= 1000:
                            val = f"{val:,.0f}"
                        else:
                            val = str(val)
                    else:
                        val = str(val) if val is not None else ""
                    values.append(val)
                lines.append("| " + " | ".join(values) + " |")
            if len(rows) > 50:
                lines.append(f"\n*... showing 50 of {len(rows)} rows*")
            return "\n".join(lines)
        else:
            lines = []
            if col_names:
                lines.append(" | ".join(col_names))
                lines.append("-" * 60)
            for r in rows[:50]:
                try:
                    lines.append(" | ".join([str(x) for x in r]))
                except Exception:
                    lines.append(str(r))
            if len(rows) > 50:
                lines.append(f"... showing 50 of {len(rows)} rows")
            return "\n".join(lines)

    if isinstance(result, list):
        return str(result[:50])[:4000]

    return str(result)[:4000]


def _make_sql_tool(name: str, description: str, query: str):
    if not AIDP_AVAILABLE:
        raise RuntimeError("SQLTool unavailable (aidputils import failed). Run `status` to see details.")

    conf = AIDPToolConf(
        name=name,
        description=description,
        tool_class="SQLTool",
        conf={
            "catalogKey": CATALOG_KEY,
            "schemaKey": SCHEMA_KEY,
            "query": query,
        },
        params=[],
    )
    return create_langgraph_tool(conf.model_dump())


# ============================================================
# Natural Language Intent Understanding
# ============================================================
SPEND_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the orchestration brain for a Spend Optimization Agent.
Your job is to understand what the user is asking about their procurement spend data and route to the right analysis.

Available analyses:
- top_suppliers: Show top suppliers by spend amount
- maverick_spend: Find purchases without contracts or unmatched suppliers
- spend_leakage: Find suppliers used across multiple ERP systems (consolidation opportunities)
- spend_by_category: Breakdown spend by category with off-contract percentages
- savings_opportunities: Find price variances for same supplier across business units
- supplier_risk: Analyze concentration risk and single-source categories
- spend_trend: Show spend trends over time by month
- sample_data: Show sample raw data
- help: Show available commands

Output STRICT JSON only (no markdown, no extra text) with this schema:
{{"intent": "<one of the analyses above>", "explanation": "Brief explanation of why this analysis answers their question", "follow_up": "Optional follow-up question to ask user, or empty string"}}

Examples:
- "Who are my biggest vendors?" -> {{"intent": "top_suppliers", "explanation": "Showing your highest-spend suppliers", "follow_up": ""}}
- "Where am I at risk?" -> {{"intent": "supplier_risk", "explanation": "Analyzing supplier concentration and single-source risks", "follow_up": ""}}
- "Show me spend without contracts" -> {{"intent": "maverick_spend", "explanation": "Finding purchases that lack contract coverage", "follow_up": ""}}
- "Where can I save money?" -> {{"intent": "savings_opportunities", "explanation": "Looking for price variances where you pay different rates for same supplier", "follow_up": ""}}
- "How is my spend trending?" -> {{"intent": "spend_trend", "explanation": "Showing monthly spend patterns over time", "follow_up": ""}}
- "What should I consolidate?" -> {{"intent": "spend_leakage", "explanation": "Finding suppliers used across multiple systems that could be consolidated", "follow_up": ""}}
- "Show me category breakdown" -> {{"intent": "spend_by_category", "explanation": "Breaking down spend by category with compliance metrics", "follow_up": ""}}

If the question is ambiguous or you are unsure, pick the most likely intent and explain your reasoning."""),
    
    ("user", "{user_message}")
])


def _parse_intent_response(raw: str) -> dict:
    """Parse LLM JSON response, handling markdown code blocks."""
    text = raw.strip()
    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"```json?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from response
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return {"intent": "help", "explanation": "I couldn't understand that. Here are the available commands.", "follow_up": ""}


# ============================================================
# Nodes
# ============================================================
def log_user_message(state: MessagesState):
    print("User asked:", _extract_text_from_last_message(state))
    return state


def status_node(state: MessagesState):
    if IMPORT_ERRORS:
        msg = "IMPORT ERRORS:\n" + "\n".join([f"- {x}" for x in IMPORT_ERRORS])
    else:
        msg = "Status OK: imports succeeded."
    msg += (
        f"\n\nCATALOG_KEY={CATALOG_KEY}"
        f"\nSCHEMA_KEY={SCHEMA_KEY}"
        f"\nSPEND_TABLE={SPEND_TABLE}"
    )
    return {"messages": [{"role": "ai", "content": msg}]}


def greet_node(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "Hello. How can I help you?"}]}


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
    user_text = _extract_text_from_last_message(state)
    return {"messages": [{"role": "ai", "content": f"You asked: '{user_text}'. Current time is {_now_utc()}"}]}


async def test_db_node(state: MessagesState):
    """Test DB connectivity with SELECT 1 FROM DUAL."""
    try:
        sql = "SELECT 1 AS test_val FROM DUAL"
        print(f"DEBUG executing SQL: {sql}")
        tool = _make_sql_tool(
            name="test_db",
            description="Test database connectivity",
            query=sql,
        )
        result = await tool.ainvoke({})
        return {"messages": [{"role": "ai", "content": "✅ DB Test Successful:\n" + _format_sqltool_result(result)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("test_db_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"❌ DB Test Failed: {e}\n\nTRACE:\n{tb}"}]}


async def sample_spend_node(state: MessagesState):
    """Primary validation: select 5 rows from the spend table."""
    try:
        # Use table name only — no quotes so Oracle uses default uppercase
        sql = f'SELECT * FROM {SPEND_TABLE} FETCH FIRST 5 ROWS ONLY'
        print(f"DEBUG executing SQL: {sql}")
        tool = _make_sql_tool(
            name="sample_spend",
            description="Fetch 5 rows from the spend table",
            query=sql,
        )
        result = await tool.ainvoke({})
        return {"messages": [{"role": "ai", "content": "Sample spend (SQLTool):\n" + _format_sqltool_result(result)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("sample_spend_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"Sample spend query failed: {e}\n\nTRACE:\n{tb}"}]}


async def top_suppliers_node(state: MessagesState):
    """Aggregate spend by canonical supplier with visual bar chart."""
    try:
        # No quotes — Oracle uses default uppercase
        sql = f'''SELECT CANONICAL_SUPPLIER_NAME AS supplier, SUM(LINE_AMOUNT_USD) AS spend_usd
FROM {SPEND_TABLE}
WHERE CANONICAL_SUPPLIER_NAME IS NOT NULL
GROUP BY CANONICAL_SUPPLIER_NAME
ORDER BY spend_usd DESC
FETCH FIRST 10 ROWS ONLY'''

        print(f"DEBUG executing SQL: {sql}")
        
        tool = _make_sql_tool(
            name="top_suppliers",
            description="Top suppliers by spend",
            query=sql,
        )
        result = await tool.ainvoke({})
        rows = _extract_rows_from_result(result)
        
        if not rows:
            return {"messages": [{"role": "ai", "content": "No supplier data found. Try `sample spend` to verify data exists."}]}
        
        # Calculate total and max for context
        total_spend = sum(float(r.get("SPEND_USD", 0) or 0) for r in rows)
        max_spend = max(float(r.get("SPEND_USD", 0) or 0) for r in rows) if rows else 0
        
        # Build formatted output with bar chart
        lines = []
        lines.append("📊 **Top 10 Suppliers by Spend**\n")
        lines.append(f"Total across top 10: {_format_currency(total_spend)}\n")
        lines.append("| Rank | Supplier | Spend | |")
        lines.append("| --- | --- | --- | --- |")
        
        for i, row in enumerate(rows, 1):
            supplier = row.get("SUPPLIER", row.get("supplier", "Unknown"))
            spend = float(row.get("SPEND_USD", row.get("spend_usd", 0)) or 0)
            bar = _make_bar(spend, max_spend, 10, "blue")
            lines.append(f"| {i} | {supplier} | {_format_currency(spend)} | {bar} |")
        
        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("top_suppliers_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"Top suppliers query failed: {e}\n\nTRACE:\n{tb}"}]}


async def create_spend_table_node(state: MessagesState):
    """Create the harmonized_spend table in the External catalog."""
    try:
        # Oracle DDL for harmonized_spend table
        sql = """
CREATE TABLE harmonized_spend (
    po_id VARCHAR2(100),
    po_line_id VARCHAR2(100),
    supplier_id VARCHAR2(100),
    supplier_name VARCHAR2(500),
    category_id VARCHAR2(100),
    category_name VARCHAR2(500),
    business_unit VARCHAR2(200),
    cost_center VARCHAR2(200),
    po_date DATE,
    currency_code VARCHAR2(10),
    line_amount_txn NUMBER(18,2),
    line_amount_usd NUMBER(18,2),
    contract_reference VARCHAR2(200),
    po_status VARCHAR2(50),
    buyer_name VARCHAR2(200),
    item_description VARCHAR2(2000),
    source_system VARCHAR2(100),
    global_supplier_id VARCHAR2(100),
    canonical_supplier_name VARCHAR2(500),
    match_confidence NUMBER(5,2),
    match_method VARCHAR2(100)
)
"""
        print(f"DEBUG executing DDL: CREATE TABLE harmonized_spend...")
        tool = _make_sql_tool(
            name="create_spend_table",
            description="Create harmonized_spend table",
            query=sql,
        )
        result = await tool.ainvoke({})
        return {"messages": [{"role": "ai", "content": "✅ CREATE TABLE harmonized_spend executed:\n" + _format_sqltool_result(result)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("create_spend_table_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"❌ CREATE TABLE failed: {e}\n\nTRACE:\n{tb}"}]}


async def insert_test_data_node(state: MessagesState):
    """Insert synthetic test data into HARMONIZED_SPEND table."""
    try:
        # Realistic multi-source spend data for POC demo
        sql = """
INSERT ALL
    INTO HARMONIZED_SPEND (po_id, po_line_id, supplier_id, supplier_name, category_id, category_name, business_unit, cost_center, po_date, currency_code, line_amount_txn, line_amount_usd, contract_reference, po_status, buyer_name, item_description, source_system, global_supplier_id, canonical_supplier_name, match_confidence, match_method)
    VALUES ('PO-2024-001', 'L001', 'SUP001', 'Acme Corp', 'CAT01', 'IT Hardware', 'Corporate IT', 'CC-1001', DATE '2024-01-15', 'USD', 45000.00, 45000.00, 'CTR-2024-100', 'APPROVED', 'John Smith', 'Dell Laptops x50', 'ORACLE_ERP', 'GSUP-001', 'Acme Corporation', 95.5, 'FUZZY_MATCH')
    INTO HARMONIZED_SPEND (po_id, po_line_id, supplier_id, supplier_name, category_id, category_name, business_unit, cost_center, po_date, currency_code, line_amount_txn, line_amount_usd, contract_reference, po_status, buyer_name, item_description, source_system, global_supplier_id, canonical_supplier_name, match_confidence, match_method)
    VALUES ('PO-2024-002', 'L001', 'SUP002', 'Acme Corporation Inc', 'CAT01', 'IT Hardware', 'Corporate IT', 'CC-1001', DATE '2024-01-20', 'USD', 32000.00, 32000.00, 'CTR-2024-100', 'APPROVED', 'John Smith', 'HP Monitors x100', 'SAP_EBS', 'GSUP-001', 'Acme Corporation', 92.0, 'FUZZY_MATCH')
    INTO HARMONIZED_SPEND (po_id, po_line_id, supplier_id, supplier_name, category_id, category_name, business_unit, cost_center, po_date, currency_code, line_amount_txn, line_amount_usd, contract_reference, po_status, buyer_name, item_description, source_system, global_supplier_id, canonical_supplier_name, match_confidence, match_method)
    VALUES ('PO-2024-003', 'L001', 'SUP003', 'TechSupply Ltd', 'CAT02', 'Software Licenses', 'Engineering', 'CC-2001', DATE '2024-02-01', 'EUR', 75000.00, 81000.00, 'CTR-2024-101', 'APPROVED', 'Jane Doe', 'Microsoft 365 Enterprise', 'ORACLE_ERP', 'GSUP-002', 'TechSupply Limited', 100.0, 'EXACT_MATCH')
    INTO HARMONIZED_SPEND (po_id, po_line_id, supplier_id, supplier_name, category_id, category_name, business_unit, cost_center, po_date, currency_code, line_amount_txn, line_amount_usd, contract_reference, po_status, buyer_name, item_description, source_system, global_supplier_id, canonical_supplier_name, match_confidence, match_method)
    VALUES ('PO-2024-004', 'L001', 'SUP004', 'Global Office Supplies', 'CAT03', 'Office Supplies', 'HR', 'CC-3001', DATE '2024-02-10', 'USD', 5200.00, 5200.00, NULL, 'APPROVED', 'Mike Johnson', 'Office furniture', 'SAP_EBS', 'GSUP-003', 'Global Office Supplies Inc', 88.5, 'FUZZY_MATCH')
    INTO HARMONIZED_SPEND (po_id, po_line_id, supplier_id, supplier_name, category_id, category_name, business_unit, cost_center, po_date, currency_code, line_amount_txn, line_amount_usd, contract_reference, po_status, buyer_name, item_description, source_system, global_supplier_id, canonical_supplier_name, match_confidence, match_method)
    VALUES ('PO-2024-005', 'L001', 'SUP005', 'CloudServ Inc', 'CAT02', 'Software Licenses', 'Engineering', 'CC-2001', DATE '2024-02-15', 'USD', 120000.00, 120000.00, 'CTR-2024-102', 'APPROVED', 'Jane Doe', 'AWS Reserved Instances', 'ORACLE_ERP', 'GSUP-004', 'CloudServ Incorporated', 100.0, 'EXACT_MATCH')
    INTO HARMONIZED_SPEND (po_id, po_line_id, supplier_id, supplier_name, category_id, category_name, business_unit, cost_center, po_date, currency_code, line_amount_txn, line_amount_usd, contract_reference, po_status, buyer_name, item_description, source_system, global_supplier_id, canonical_supplier_name, match_confidence, match_method)
    VALUES ('PO-2024-006', 'L001', 'SUP001', 'ACME CORP', 'CAT01', 'IT Hardware', 'Sales', 'CC-4001', DATE '2024-02-20', 'USD', 28000.00, 28000.00, 'CTR-2024-100', 'PENDING', 'Sarah Lee', 'Network switches x20', 'ARIBA', 'GSUP-001', 'Acme Corporation', 90.0, 'FUZZY_MATCH')
    INTO HARMONIZED_SPEND (po_id, po_line_id, supplier_id, supplier_name, category_id, category_name, business_unit, cost_center, po_date, currency_code, line_amount_txn, line_amount_usd, contract_reference, po_status, buyer_name, item_description, source_system, global_supplier_id, canonical_supplier_name, match_confidence, match_method)
    VALUES ('PO-2024-007', 'L001', 'SUP006', 'FastShip Logistics', 'CAT04', 'Logistics', 'Operations', 'CC-5001', DATE '2024-03-01', 'USD', 15500.00, 15500.00, NULL, 'APPROVED', 'Tom Brown', 'Q1 shipping services', 'SAP_EBS', 'GSUP-005', 'FastShip Logistics LLC', 100.0, 'EXACT_MATCH')
    INTO HARMONIZED_SPEND (po_id, po_line_id, supplier_id, supplier_name, category_id, category_name, business_unit, cost_center, po_date, currency_code, line_amount_txn, line_amount_usd, contract_reference, po_status, buyer_name, item_description, source_system, global_supplier_id, canonical_supplier_name, match_confidence, match_method)
    VALUES ('PO-2024-008', 'L001', 'SUP007', 'Premier Consulting', 'CAT05', 'Professional Services', 'Finance', 'CC-6001', DATE '2024-03-05', 'USD', 95000.00, 95000.00, 'CTR-2024-103', 'APPROVED', 'Lisa Wang', 'Financial audit Q1', 'ORACLE_ERP', 'GSUP-006', 'Premier Consulting Group', 97.0, 'FUZZY_MATCH')
    INTO HARMONIZED_SPEND (po_id, po_line_id, supplier_id, supplier_name, category_id, category_name, business_unit, cost_center, po_date, currency_code, line_amount_txn, line_amount_usd, contract_reference, po_status, buyer_name, item_description, source_system, global_supplier_id, canonical_supplier_name, match_confidence, match_method)
    VALUES ('PO-2024-009', 'L001', 'SUP008', 'TechSupply Limited', 'CAT02', 'Software Licenses', 'Corporate IT', 'CC-1001', DATE '2024-03-10', 'GBP', 42000.00, 53000.00, 'CTR-2024-101', 'APPROVED', 'John Smith', 'Salesforce licenses', 'ARIBA', 'GSUP-002', 'TechSupply Limited', 100.0, 'EXACT_MATCH')
    INTO HARMONIZED_SPEND (po_id, po_line_id, supplier_id, supplier_name, category_id, category_name, business_unit, cost_center, po_date, currency_code, line_amount_txn, line_amount_usd, contract_reference, po_status, buyer_name, item_description, source_system, global_supplier_id, canonical_supplier_name, match_confidence, match_method)
    VALUES ('PO-2024-010', 'L001', 'SUP009', 'Maverick Supplies', 'CAT03', 'Office Supplies', 'Marketing', 'CC-7001', DATE '2024-03-15', 'USD', 3200.00, 3200.00, NULL, 'APPROVED', 'Chris Martin', 'Marketing materials', 'MANUAL', NULL, NULL, NULL, 'NO_MATCH')
    INTO HARMONIZED_SPEND (po_id, po_line_id, supplier_id, supplier_name, category_id, category_name, business_unit, cost_center, po_date, currency_code, line_amount_txn, line_amount_usd, contract_reference, po_status, buyer_name, item_description, source_system, global_supplier_id, canonical_supplier_name, match_confidence, match_method)
    VALUES ('PO-2024-011', 'L001', 'SUP010', 'CloudServ', 'CAT02', 'Software Licenses', 'Engineering', 'CC-2001', DATE '2024-03-20', 'USD', 85000.00, 85000.00, 'CTR-2024-102', 'APPROVED', 'Jane Doe', 'GCP compute credits', 'SAP_EBS', 'GSUP-004', 'CloudServ Incorporated', 88.0, 'FUZZY_MATCH')
    INTO HARMONIZED_SPEND (po_id, po_line_id, supplier_id, supplier_name, category_id, category_name, business_unit, cost_center, po_date, currency_code, line_amount_txn, line_amount_usd, contract_reference, po_status, buyer_name, item_description, source_system, global_supplier_id, canonical_supplier_name, match_confidence, match_method)
    VALUES ('PO-2024-012', 'L001', 'SUP011', 'Quick Print Services', 'CAT06', 'Marketing', 'Marketing', 'CC-7001', DATE '2024-03-25', 'USD', 8500.00, 8500.00, NULL, 'PENDING', 'Chris Martin', 'Brochures and banners', 'MANUAL', NULL, NULL, NULL, 'NO_MATCH')
SELECT 1 FROM DUAL
"""
        print(f"DEBUG executing INSERT ALL...")
        tool = _make_sql_tool(
            name="insert_test_data",
            description="Insert synthetic spend data",
            query=sql,
        )
        result = await tool.ainvoke({})
        return {"messages": [{"role": "ai", "content": "✅ Inserted 12 test records into HARMONIZED_SPEND:\n" + _format_sqltool_result(result) + "\n\nTry `sample spend` or `top suppliers` to see the data."}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("insert_test_data_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"❌ INSERT failed: {e}\n\nTRACE:\n{tb}"}]}


async def maverick_spend_node(state: MessagesState):
    """Detect maverick spend - purchases without contracts or unmatched suppliers."""
    try:
        sql = f'''SELECT 
    PO_ID,
    SUPPLIER_NAME,
    CATEGORY_NAME,
    BUSINESS_UNIT,
    LINE_AMOUNT_USD,
    CONTRACT_REFERENCE,
    MATCH_METHOD,
    CASE 
        WHEN CONTRACT_REFERENCE IS NULL AND MATCH_METHOD = 'NO_MATCH' THEN 'No Contract + Unmatched Supplier'
        WHEN CONTRACT_REFERENCE IS NULL THEN 'No Contract'
        WHEN MATCH_METHOD = 'NO_MATCH' THEN 'Unmatched Supplier'
        ELSE 'Low Confidence Match'
    END AS RISK_TYPE
FROM {SPEND_TABLE}
WHERE CONTRACT_REFERENCE IS NULL 
   OR MATCH_METHOD = 'NO_MATCH'
   OR MATCH_CONFIDENCE < 90
ORDER BY LINE_AMOUNT_USD DESC
FETCH FIRST 20 ROWS ONLY'''

        print(f"DEBUG executing maverick spend SQL...")
        tool = _make_sql_tool(
            name="maverick_spend",
            description="Detect maverick spend",
            query=sql,
        )
        result = await tool.ainvoke({})
        rows = _extract_rows_from_result(result)
        
        if not rows:
            return {"messages": [{"role": "ai", "content": "✅ No maverick spend detected. All transactions have contracts and matched suppliers."}]}
        
        # Calculate summary stats
        total_maverick = sum(float(r.get("LINE_AMOUNT_USD", 0) or 0) for r in rows)
        no_contract_rows = [r for r in rows if "No Contract" in str(r.get("RISK_TYPE", ""))]
        no_contract_total = sum(float(r.get("LINE_AMOUNT_USD", 0) or 0) for r in no_contract_rows)
        unmatched_count = sum(1 for r in rows if "Unmatched" in str(r.get("RISK_TYPE", "")))
        low_conf_count = sum(1 for r in rows if "Low Confidence" in str(r.get("RISK_TYPE", "")))
        
        # Find top offenders for recommendations
        top_supplier = rows[0].get("SUPPLIER_NAME", "Unknown") if rows else "Unknown"
        top_amount = float(rows[0].get("LINE_AMOUNT_USD", 0) or 0) if rows else 0
        
        # Group by supplier to find repeat offenders
        supplier_counts = {}
        supplier_totals = {}
        for r in rows:
            sup = r.get("SUPPLIER_NAME", "Unknown")
            amt = float(r.get("LINE_AMOUNT_USD", 0) or 0)
            supplier_counts[sup] = supplier_counts.get(sup, 0) + 1
            supplier_totals[sup] = supplier_totals.get(sup, 0) + amt
        repeat_offenders = [(s, supplier_counts[s], supplier_totals[s]) for s in supplier_counts if supplier_counts[s] > 1]
        repeat_offenders.sort(key=lambda x: x[2], reverse=True)
        
        # Build formatted output
        lines = []
        lines.append("🚨 **Maverick Spend Analysis**\n")
        lines.append(f"**Total Flagged:** {_format_currency(total_maverick)} across {len(rows)} transactions\n")
        lines.append("**Breakdown:**")
        if no_contract_rows:
            lines.append(f"  🟠 No Contract: {len(no_contract_rows)} transactions ({_format_currency(no_contract_total)})")
        if unmatched_count > 0:
            lines.append(f"  🟠 Unmatched Supplier: {unmatched_count} transactions")
        if low_conf_count > 0:
            lines.append(f"  🟡 Low Confidence Match: {low_conf_count} transactions")
        lines.append("")
        
        # Table
        lines.append("| Risk | Supplier | Category | Amount |")
        lines.append("| --- | --- | --- | --- |")
        
        for row in rows[:15]:
            risk = row.get("RISK_TYPE", "")
            emoji = _risk_emoji(risk)
            supplier = row.get("SUPPLIER_NAME", "Unknown")[:25]
            category = row.get("CATEGORY_NAME", "")[:20]
            amount = float(row.get("LINE_AMOUNT_USD", 0) or 0)
            lines.append(f"| {emoji} | {supplier} | {category} | {_format_currency(amount)} |")
        
        if len(rows) > 15:
            lines.append(f"\n*Showing top 15 of {len(rows)} flagged transactions*")
        
        # Smart recommendations based on actual data
        lines.append("\n---\n**🎯 Recommended Actions:**\n")
        
        # Priority 1: Highest value item
        lines.append(f"**1. Immediate:** Review {top_supplier} ({_format_currency(top_amount)}) — your largest no-contract purchase. Determine if this should be under a master agreement.")
        
        # Priority 2: Repeat offenders
        if repeat_offenders:
            top_repeat = repeat_offenders[0]
            lines.append(f"\n**2. Quick Win:** {top_repeat[0]} has {top_repeat[1]} flagged transactions totaling {_format_currency(top_repeat[2])}. Negotiate a blanket contract to cover all purchases.")
        
        # Priority 3: Process improvement
        if no_contract_total > 100000:
            lines.append(f"\n**3. Process Fix:** {_format_currency(no_contract_total)} in no-contract spend suggests a procurement policy gap. Consider requiring contract reference for POs over $10K.")
        
        # Next step
        lines.append(f"\n**Next:** Run `supplier risk` to check if any of these suppliers represent concentration risk.")
        
        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("maverick_spend_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"Maverick spend analysis failed: {e}\n\nTRACE:\n{tb}"}]}


async def spend_leakage_node(state: MessagesState):
    """Identify spend leakage - consolidation opportunities across source systems."""
    try:
        sql = f'''SELECT 
    CANONICAL_SUPPLIER_NAME,
    COUNT(DISTINCT SOURCE_SYSTEM) AS NUM_SYSTEMS,
    LISTAGG(DISTINCT SOURCE_SYSTEM, ', ') WITHIN GROUP (ORDER BY SOURCE_SYSTEM) AS SYSTEMS,
    COUNT(*) AS TRANSACTION_COUNT,
    SUM(LINE_AMOUNT_USD) AS TOTAL_SPEND,
    ROUND(AVG(MATCH_CONFIDENCE), 1) AS AVG_CONFIDENCE
FROM {SPEND_TABLE}
WHERE CANONICAL_SUPPLIER_NAME IS NOT NULL
GROUP BY CANONICAL_SUPPLIER_NAME
HAVING COUNT(DISTINCT SOURCE_SYSTEM) > 1
ORDER BY TOTAL_SPEND DESC
FETCH FIRST 15 ROWS ONLY'''

        print(f"DEBUG executing spend leakage SQL...")
        tool = _make_sql_tool(
            name="spend_leakage",
            description="Identify spend consolidation opportunities",
            query=sql,
        )
        result = await tool.ainvoke({})
        rows = _extract_rows_from_result(result)
        
        if not rows:
            return {"messages": [{"role": "ai", "content": "✅ No spend leakage detected. Each supplier is being managed through a single system."}]}
        
        # Calculate summary stats
        total_leakage = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows)
        max_spend = max(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows) if rows else 0
        
        # Find top opportunities
        top3 = rows[:3]
        top3_spend = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in top3)
        
        # Find any with 3+ systems (highest priority)
        multi_system = [r for r in rows if int(r.get("NUM_SYSTEMS", 0) or 0) >= 3]
        
        # Build formatted output
        lines = []
        lines.append("💰 **Spend Consolidation Opportunities**\n")
        lines.append(f"**Total Addressable Spend:** {_format_currency(total_leakage)} across {len(rows)} suppliers\n")
        lines.append("Suppliers being used across multiple ERP systems — opportunity for contract consolidation:\n")
        
        # Table with bar chart
        lines.append("| Supplier | Systems | Spend | |")
        lines.append("| --- | --- | --- | --- |")
        
        for row in rows:
            supplier = str(row.get("CANONICAL_SUPPLIER_NAME", "Unknown"))[:25]
            num_sys = row.get("NUM_SYSTEMS", 0)
            spend = float(row.get("TOTAL_SPEND", 0) or 0)
            bar = _make_bar(spend, max_spend, 8, "green")
            
            sys_indicator = "🔴" if num_sys >= 3 else "🟡"
            lines.append(f"| {supplier} | {sys_indicator} {num_sys} | {_format_currency(spend)} | {bar} |")
        
        lines.append("")
        lines.append("**Legend:** 🔴 3+ systems (high priority) | 🟡 2 systems")
        
        # Smart recommendations based on actual data
        lines.append("\n---\n**🎯 Recommended Actions:**\n")
        
        # Priority 1: Top opportunity
        if rows:
            top = rows[0]
            top_name = top.get("CANONICAL_SUPPLIER_NAME", "Unknown")
            top_spend = float(top.get("TOTAL_SPEND", 0) or 0)
            top_systems = top.get("SYSTEMS", "")
            est_savings = top_spend * 0.08  # Assume 8% savings from consolidation
            lines.append(f"**1. Biggest Opportunity:** {top_name} ({_format_currency(top_spend)} across {top_systems})")
            lines.append(f"   → Consolidate under single contract. Est. savings: {_format_currency(est_savings)} (8% volume discount)")
        
        # Priority 2: Multi-system suppliers
        if multi_system:
            lines.append(f"\n**2. High Complexity:** {len(multi_system)} suppliers span 3+ systems. These create the most overhead:")
            for r in multi_system[:3]:
                lines.append(f"   • {r.get('CANONICAL_SUPPLIER_NAME', 'Unknown')}: {r.get('SYSTEMS', '')}")
        
        # Priority 3: Quick wins
        if len(rows) >= 3:
            lines.append(f"\n**3. Quick Win:** Consolidating just your top 3 suppliers addresses {_format_currency(top3_spend)} ({top3_spend/total_leakage*100:.0f}% of leakage).")
        
        # Estimated total impact
        total_est_savings = total_leakage * 0.05  # Conservative 5%
        lines.append(f"\n**💵 Total Savings Potential:** {_format_currency(total_est_savings)} (5% of addressable spend through consolidation)")
        
        # Next step
        lines.append(f"\n**Next:** Run `savings opportunities` to find price variances for these same suppliers.")
        
        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("spend_leakage_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"Spend leakage analysis failed: {e}\n\nTRACE:\n{tb}"}]}


async def spend_by_category_node(state: MessagesState):
    """Analyze spend by category with off-contract risk."""
    try:
        sql = f'''SELECT 
    CATEGORY_NAME,
    COUNT(*) AS TRANSACTION_COUNT,
    SUM(LINE_AMOUNT_USD) AS TOTAL_SPEND,
    COUNT(DISTINCT CANONICAL_SUPPLIER_NAME) AS UNIQUE_SUPPLIERS,
    ROUND(SUM(CASE WHEN CONTRACT_REFERENCE IS NULL THEN LINE_AMOUNT_USD ELSE 0 END) / 
          NULLIF(SUM(LINE_AMOUNT_USD), 0) * 100, 1) AS PCT_OFF_CONTRACT
FROM {SPEND_TABLE}
GROUP BY CATEGORY_NAME
ORDER BY TOTAL_SPEND DESC'''

        print(f"DEBUG executing spend by category SQL...")
        tool = _make_sql_tool(
            name="spend_by_category",
            description="Analyze spend by category",
            query=sql,
        )
        result = await tool.ainvoke({})
        rows = _extract_rows_from_result(result)
        
        if not rows:
            return {"messages": [{"role": "ai", "content": "No category data found."}]}
        
        # Calculate summary stats
        total_spend = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows)
        max_spend = max(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows) if rows else 0
        high_risk_categories = [r for r in rows if float(r.get("PCT_OFF_CONTRACT", 0) or 0) > 50]
        
        # Build formatted output
        lines = []
        lines.append("📊 **Spend by Category**\n")
        lines.append(f"**Total Spend:** {_format_currency(total_spend)} across {len(rows)} categories")
        if high_risk_categories:
            lines.append(f"**⚠️ High Risk:** {len(high_risk_categories)} categories with >50% off-contract spend\n")
        else:
            lines.append("")
        
        # Table with bar chart and risk indicators
        lines.append("| Category | Spend | | Off-Contract | Risk |")
        lines.append("| --- | --- | --- | --- | --- |")
        
        for row in rows:
            category = str(row.get("CATEGORY_NAME", "Unknown"))[:25]
            spend = float(row.get("TOTAL_SPEND", 0) or 0)
            bar = _make_bar(spend, max_spend, 8, "purple")
            pct_off = float(row.get("PCT_OFF_CONTRACT", 0) or 0)
            
            # Risk indicator based on off-contract percentage
            if pct_off >= 75:
                risk = "🔴 High"
            elif pct_off >= 50:
                risk = "🟠 Medium"
            elif pct_off >= 25:
                risk = "🟡 Low"
            else:
                risk = "🟢 OK"
            
            lines.append(f"| {category} | {_format_currency(spend)} | {bar} | {pct_off:.0f}% | {risk} |")
        
        lines.append("")
        lines.append("**Legend:** Risk based on % of spend without contracts")
        
        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("spend_by_category_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"Spend by category analysis failed: {e}\n\nTRACE:\n{tb}"}]}


async def savings_opportunities_node(state: MessagesState):
    """Identify price variance savings - same supplier, different prices across BUs."""
    try:
        sql = f'''SELECT 
    CANONICAL_SUPPLIER_NAME,
    CATEGORY_NAME,
    COUNT(DISTINCT BUSINESS_UNIT) AS NUM_BUS,
    MIN(LINE_AMOUNT_USD) AS MIN_PRICE,
    MAX(LINE_AMOUNT_USD) AS MAX_PRICE,
    ROUND(AVG(LINE_AMOUNT_USD), 2) AS AVG_PRICE,
    ROUND((MAX(LINE_AMOUNT_USD) - MIN(LINE_AMOUNT_USD)) / NULLIF(AVG(LINE_AMOUNT_USD), 0) * 100, 1) AS VARIANCE_PCT,
    SUM(LINE_AMOUNT_USD) AS TOTAL_SPEND,
    COUNT(*) AS TRANSACTION_COUNT
FROM {SPEND_TABLE}
WHERE CANONICAL_SUPPLIER_NAME IS NOT NULL
GROUP BY CANONICAL_SUPPLIER_NAME, CATEGORY_NAME
HAVING COUNT(DISTINCT BUSINESS_UNIT) > 1 
   AND MAX(LINE_AMOUNT_USD) > MIN(LINE_AMOUNT_USD) * 1.2
ORDER BY (MAX(LINE_AMOUNT_USD) - MIN(LINE_AMOUNT_USD)) * COUNT(*) DESC
FETCH FIRST 15 ROWS ONLY'''

        print(f"DEBUG executing savings opportunities SQL...")
        tool = _make_sql_tool(
            name="savings_opportunities",
            description="Identify price variance savings opportunities",
            query=sql,
        )
        result = await tool.ainvoke({})
        rows = _extract_rows_from_result(result)
        
        if not rows:
            return {"messages": [{"role": "ai", "content": "✅ No significant price variances detected. Pricing appears consistent across business units."}]}
        
        # Calculate potential savings
        total_addressable = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows)
        
        # Calculate realistic savings: for each row, savings = (max - min) / 2 * transaction_count
        est_savings = 0
        for r in rows:
            min_p = float(r.get("MIN_PRICE", 0) or 0)
            max_p = float(r.get("MAX_PRICE", 0) or 0)
            txn_count = int(r.get("TRANSACTION_COUNT", 0) or 0)
            # If we normalize to midpoint, we save half the variance per transaction
            est_savings += ((max_p - min_p) / 2) * (txn_count / 2)  # Conservative
        
        # Find top opportunities
        top_row = rows[0] if rows else {}
        top_supplier = top_row.get("CANONICAL_SUPPLIER_NAME", "Unknown")
        top_category = top_row.get("CATEGORY_NAME", "Unknown")
        top_variance = float(top_row.get("VARIANCE_PCT", 0) or 0)
        top_min = float(top_row.get("MIN_PRICE", 0) or 0)
        top_max = float(top_row.get("MAX_PRICE", 0) or 0)
        
        # Find highest variance
        highest_var_row = max(rows, key=lambda r: float(r.get("VARIANCE_PCT", 0) or 0))
        highest_var = float(highest_var_row.get("VARIANCE_PCT", 0) or 0)
        highest_var_supplier = highest_var_row.get("CANONICAL_SUPPLIER_NAME", "Unknown")
        
        lines = []
        lines.append("💵 **Savings Opportunities — Price Variance Analysis**\n")
        lines.append(f"**Addressable Spend:** {_format_currency(total_addressable)}")
        lines.append(f"**Estimated Savings Potential:** {_format_currency(est_savings)}\n")
        lines.append("Same supplier charging different prices across business units:\n")
        
        lines.append("| Supplier | Category | BUs | Min | Max | Variance |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        
        for row in rows[:12]:
            supplier = str(row.get("CANONICAL_SUPPLIER_NAME", ""))[:20]
            category = str(row.get("CATEGORY_NAME", ""))[:18]
            num_bus = row.get("NUM_BUS", 0)
            min_price = float(row.get("MIN_PRICE", 0) or 0)
            max_price = float(row.get("MAX_PRICE", 0) or 0)
            variance = float(row.get("VARIANCE_PCT", 0) or 0)
            
            if variance >= 50:
                var_display = f"🔴 {variance:.0f}%"
            elif variance >= 30:
                var_display = f"🟠 {variance:.0f}%"
            else:
                var_display = f"🟡 {variance:.0f}%"
            
            lines.append(f"| {supplier} | {category} | {num_bus} | {_format_currency(min_price)} | {_format_currency(max_price)} | {var_display} |")
        
        # Smart recommendations
        lines.append("\n---\n**🎯 Recommended Actions:**\n")
        
        # Priority 1: Biggest impact opportunity
        lines.append(f"**1. Start Here:** {top_supplier} / {top_category}")
        lines.append(f"   → Price ranges from {_format_currency(top_min)} to {_format_currency(top_max)} ({top_variance:.0f}% variance)")
        lines.append(f"   → Action: Contact supplier to negotiate standardized pricing at {_format_currency((top_min + top_max) / 2)} or lower")
        
        # Priority 2: Highest variance (might be different from biggest spend)
        if highest_var_supplier != top_supplier and highest_var > 100:
            lines.append(f"\n**2. Biggest Variance:** {highest_var_supplier} has {highest_var:.0f}% price variance")
            lines.append(f"   → This suggests ad-hoc purchasing or missing master agreement")
        
        # Priority 3: Process improvement
        lines.append(f"\n**3. Process Fix:** Implement standard pricing sheets for top 10 suppliers. Require PO approval if price exceeds catalog by >10%.")
        
        # Impact summary
        lines.append(f"\n**💵 Bottom Line:** Standardizing prices to midpoint could save {_format_currency(est_savings)} annually.")
        
        # Next step
        lines.append(f"\n**Next:** Run `spend leakage` to see if these same suppliers are fragmented across systems.")
        
        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("savings_opportunities_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"Savings analysis failed: {e}\n\nTRACE:\n{tb}"}]}


async def supplier_risk_node(state: MessagesState):
    """Analyze supplier concentration risk - single source categories, over-reliance."""
    try:
        # First query: Category concentration (single supplier categories)
        sql_concentration = f'''SELECT 
    CATEGORY_NAME,
    COUNT(DISTINCT CANONICAL_SUPPLIER_NAME) AS SUPPLIER_COUNT,
    SUM(LINE_AMOUNT_USD) AS TOTAL_SPEND,
    MAX(CANONICAL_SUPPLIER_NAME) AS PRIMARY_SUPPLIER
FROM {SPEND_TABLE}
WHERE CANONICAL_SUPPLIER_NAME IS NOT NULL
GROUP BY CATEGORY_NAME
HAVING COUNT(DISTINCT CANONICAL_SUPPLIER_NAME) = 1 AND SUM(LINE_AMOUNT_USD) > 100000
ORDER BY TOTAL_SPEND DESC
FETCH FIRST 10 ROWS ONLY'''

        # Second query: Supplier over-reliance (>5% of total spend)
        sql_reliance = f'''SELECT 
    CANONICAL_SUPPLIER_NAME,
    SUM(LINE_AMOUNT_USD) AS SUPPLIER_SPEND,
    COUNT(*) AS TRANSACTION_COUNT,
    COUNT(DISTINCT CATEGORY_NAME) AS CATEGORIES_SERVED,
    ROUND(SUM(LINE_AMOUNT_USD) / (SELECT SUM(LINE_AMOUNT_USD) FROM {SPEND_TABLE}) * 100, 1) AS PCT_OF_TOTAL
FROM {SPEND_TABLE}
WHERE CANONICAL_SUPPLIER_NAME IS NOT NULL
GROUP BY CANONICAL_SUPPLIER_NAME
HAVING SUM(LINE_AMOUNT_USD) / (SELECT SUM(LINE_AMOUNT_USD) FROM {SPEND_TABLE}) > 0.05
ORDER BY SUPPLIER_SPEND DESC
FETCH FIRST 10 ROWS ONLY'''

        print(f"DEBUG executing supplier risk SQL...")
        
        # Get concentration data
        tool1 = _make_sql_tool(
            name="supplier_concentration",
            description="Single supplier categories",
            query=sql_concentration,
        )
        result1 = await tool1.ainvoke({})
        concentration_rows = _extract_rows_from_result(result1)
        
        # Get reliance data
        tool2 = _make_sql_tool(
            name="supplier_reliance",
            description="Supplier over-reliance",
            query=sql_reliance,
        )
        result2 = await tool2.ainvoke({})
        reliance_rows = _extract_rows_from_result(result2)
        
        lines = []
        lines.append("⚠️ **Supplier Risk Analysis**\n")
        
        # Section 1: Single-source categories
        if concentration_rows:
            single_source_spend = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in concentration_rows)
            lines.append(f"**🔴 Single-Source Categories:** {len(concentration_rows)} categories ({_format_currency(single_source_spend)} at risk)\n")
            lines.append("| Category | Sole Supplier | Spend |")
            lines.append("| --- | --- | --- |")
            for row in concentration_rows[:6]:
                cat = str(row.get("CATEGORY_NAME", ""))[:22]
                supplier = str(row.get("PRIMARY_SUPPLIER", ""))[:22]
                spend = float(row.get("TOTAL_SPEND", 0) or 0)
                lines.append(f"| {cat} | {supplier} | {_format_currency(spend)} |")
            lines.append("")
        else:
            lines.append("✅ **Single-Source:** No high-value single-source categories detected.\n")
        
        # Section 2: Supplier over-reliance
        if reliance_rows:
            total_concentrated = sum(float(r.get("SUPPLIER_SPEND", 0) or 0) for r in reliance_rows)
            top_supplier = reliance_rows[0] if reliance_rows else {}
            top_name = top_supplier.get("CANONICAL_SUPPLIER_NAME", "Unknown")
            top_pct = float(top_supplier.get("PCT_OF_TOTAL", 0) or 0)
            
            lines.append("**🟠 Supplier Concentration:**\n")
            lines.append("| Supplier | Spend | % of Total | Categories |")
            lines.append("| --- | --- | --- | --- |")
            max_spend = max(float(r.get("SUPPLIER_SPEND", 0) or 0) for r in reliance_rows) if reliance_rows else 0
            for row in reliance_rows[:8]:
                supplier = str(row.get("CANONICAL_SUPPLIER_NAME", ""))[:22]
                spend = float(row.get("SUPPLIER_SPEND", 0) or 0)
                pct = float(row.get("PCT_OF_TOTAL", 0) or 0)
                cats = row.get("CATEGORIES_SERVED", 0)
                bar = _make_bar(spend, max_spend, 6, "orange")
                lines.append(f"| {supplier} | {_format_currency(spend)} | {pct:.1f}% {bar} | {cats} |")
            lines.append("")
        else:
            lines.append("✅ **Concentration:** No suppliers exceed 5% of total spend.\n")
        
        # Smart recommendations
        lines.append("---\n**🎯 Recommended Actions:**\n")
        
        # Priority 1: Single-source risk
        if concentration_rows:
            top_single = concentration_rows[0]
            top_cat = top_single.get("CATEGORY_NAME", "Unknown")
            top_sup = top_single.get("PRIMARY_SUPPLIER", "Unknown")
            top_spend = float(top_single.get("TOTAL_SPEND", 0) or 0)
            lines.append(f"**1. Critical Risk:** {top_cat} depends entirely on {top_sup} ({_format_currency(top_spend)})")
            lines.append(f"   → Action: Identify and qualify 1-2 alternate suppliers within 30 days")
            lines.append(f"   → Interim: Review contract terms for early termination risk")
        
        # Priority 2: Concentration leverage
        if reliance_rows:
            top_rel = reliance_rows[0]
            top_name = top_rel.get("CANONICAL_SUPPLIER_NAME", "Unknown")
            top_spend = float(top_rel.get("SUPPLIER_SPEND", 0) or 0)
            top_pct = float(top_rel.get("PCT_OF_TOTAL", 0) or 0)
            lines.append(f"\n**2. Leverage Opportunity:** {top_name} represents {top_pct:.1f}% of spend ({_format_currency(top_spend)})")
            lines.append(f"   → Action: Schedule QBR to negotiate better terms (volume discount, payment terms, SLAs)")
            lines.append(f"   → Your volume justifies preferred pricing tier")
        
        # Priority 3: Diversification
        if concentration_rows and len(concentration_rows) >= 3:
            lines.append(f"\n**3. Diversification Plan:** {len(concentration_rows)} single-source categories need attention")
            lines.append(f"   → Prioritize by spend amount and business criticality")
            lines.append(f"   → Target: No category >$500K should be single-source")
        
        # Next step
        lines.append(f"\n**Next:** Run `maverick spend` to check if any high-risk suppliers also have contract gaps.")
        
        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("supplier_risk_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"Supplier risk analysis failed: {e}\n\nTRACE:\n{tb}"}]}


async def spend_trend_node(state: MessagesState):
    """Analyze spend trends by month/quarter."""
    try:
        sql = f'''SELECT 
    TO_CHAR(PO_DATE, 'YYYY-MM') AS MONTH,
    COUNT(*) AS TRANSACTION_COUNT,
    SUM(LINE_AMOUNT_USD) AS TOTAL_SPEND,
    COUNT(DISTINCT CANONICAL_SUPPLIER_NAME) AS ACTIVE_SUPPLIERS,
    ROUND(SUM(CASE WHEN CONTRACT_REFERENCE IS NULL THEN LINE_AMOUNT_USD ELSE 0 END) / 
          NULLIF(SUM(LINE_AMOUNT_USD), 0) * 100, 1) AS PCT_MAVERICK
FROM {SPEND_TABLE}
WHERE PO_DATE IS NOT NULL
GROUP BY TO_CHAR(PO_DATE, 'YYYY-MM')
ORDER BY MONTH DESC
FETCH FIRST 12 ROWS ONLY'''

        print(f"DEBUG executing spend trend SQL...")
        tool = _make_sql_tool(
            name="spend_trend",
            description="Analyze spend trends over time",
            query=sql,
        )
        result = await tool.ainvoke({})
        rows = _extract_rows_from_result(result)
        
        if not rows:
            return {"messages": [{"role": "ai", "content": "No trend data available. Check that PO_DATE is populated."}]}
        
        # Reverse to show oldest first for trend visualization
        rows = list(reversed(rows))
        
        # Calculate summary stats
        total_spend = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows)
        avg_monthly = total_spend / len(rows) if rows else 0
        max_spend = max(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows) if rows else 0
        
        # Trend direction
        if len(rows) >= 2:
            first_half = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows[:len(rows)//2])
            second_half = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows[len(rows)//2:])
            if second_half > first_half * 1.1:
                trend = "📈 Increasing"
            elif second_half < first_half * 0.9:
                trend = "📉 Decreasing"
            else:
                trend = "➡️ Stable"
        else:
            trend = "—"
        
        lines = []
        lines.append("📈 **Spend Trend Analysis**\n")
        lines.append(f"**Period:** {rows[0].get('MONTH', '')} to {rows[-1].get('MONTH', '')}")
        lines.append(f"**Total:** {_format_currency(total_spend)} | **Avg Monthly:** {_format_currency(avg_monthly)} | **Trend:** {trend}\n")
        
        lines.append("| Month | Spend | | Maverick % |")
        lines.append("| --- | --- | --- | --- |")
        
        for row in rows:
            month = row.get("MONTH", "")
            spend = float(row.get("TOTAL_SPEND", 0) or 0)
            bar = _make_bar(spend, max_spend, 10, "blue")
            maverick_pct = float(row.get("PCT_MAVERICK", 0) or 0)
            
            mav_indicator = "🔴" if maverick_pct > 50 else "🟡" if maverick_pct > 25 else "🟢"
            lines.append(f"| {month} | {_format_currency(spend)} | {bar} | {mav_indicator} {maverick_pct:.0f}% |")
        
        lines.append("")
        lines.append("**Legend:** Maverick % = spend without contracts")
        
        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("spend_trend_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"Spend trend analysis failed: {e}\n\nTRACE:\n{tb}"}]}


async def list_tables_node(state: MessagesState):
    """List tables in the schema using Oracle syntax."""
    try:
        # Oracle way to list tables
        sql = "SELECT table_name FROM user_tables ORDER BY table_name FETCH FIRST 50 ROWS ONLY"
        print(f"DEBUG executing SQL: {sql}")
        tool = _make_sql_tool(
            name="list_tables",
            description="List tables in schema",
            query=sql,
        )
        result = await tool.ainvoke({})
        return {"messages": [{"role": "ai", "content": f"Tables in {CATALOG_KEY}.{SCHEMA_KEY}:\n" + _format_sqltool_result(result)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("list_tables_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"List tables failed: {e}\n\nTRACE:\n{tb}\n\nTip: run `test db` to validate connectivity."}]}


async def ask_node(state: MessagesState):
    """Natural language intent understanding - routes to appropriate analysis."""
    user_text = _extract_text_from_last_message(state)
    
    try:
        if not LLM_AVAILABLE:
            return {"messages": [{"role": "ai", "content": "LLM not available. Please use specific commands like `top suppliers`, `maverick spend`, etc. Type `help` for options."}]}
        
        # Get LLM to understand intent
        llm = _ensure_llm()
        chain = SPEND_INTENT_PROMPT | llm | StrOutputParser()
        
        print(f"DEBUG: Asking LLM to interpret: {user_text}")
        raw_response = await chain.ainvoke({"user_message": user_text})
        print(f"DEBUG: LLM response: {raw_response}")
        
        parsed = _parse_intent_response(raw_response)
        intent = parsed.get("intent", "help")
        explanation = parsed.get("explanation", "")
        
        # Map intent to the actual node function
        intent_map = {
            "top_suppliers": top_suppliers_node,
            "maverick_spend": maverick_spend_node,
            "spend_leakage": spend_leakage_node,
            "spend_by_category": spend_by_category_node,
            "savings_opportunities": savings_opportunities_node,
            "supplier_risk": supplier_risk_node,
            "spend_trend": spend_trend_node,
            "sample_data": sample_spend_node,
            "help": help_node,
        }
        
        if intent in intent_map:
            # Add the explanation as a prefix
            prefix = f"💡 *{explanation}*\n\n" if explanation else ""
            
            # Call the actual analysis node
            result = await intent_map[intent](state)
            
            # Prepend explanation to the result
            if result and "messages" in result and result["messages"]:
                original_content = result["messages"][0].get("content", "")
                result["messages"][0]["content"] = prefix + original_content
            
            return result
        else:
            # Fallback to help
            return await help_node(state)
            
    except Exception as e:
        tb = traceback.format_exc()
        print("ask_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"I had trouble understanding that. Try specific commands like `top suppliers` or `maverick spend`.\n\nError: {e}"}]}


# ============================================================
# Router
# ============================================================
def router(state: MessagesState) -> str:
    text = _extract_text_from_last_message(state).lower().strip()

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
    if "create spend table" in text or "create table" in text:
        return "create_spend_table_node"
    if "insert test data" in text or "load test data" in text:
        return "insert_test_data_node"
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
    # Check for question patterns or longer queries
    question_starters = ["who", "what", "where", "why", "how", "which", "show me", "tell me", "find", "analyze", "can you", "i want", "i need"]
    if any(text.startswith(q) for q in question_starters) or len(text.split()) >= 4:
        return "ask_node"

    return "help_node"


# ============================================================
# Agent
# ============================================================
class AgentBasic:
    def __init__(self) -> None:
        self.graph = None

    def setup(self) -> None:
        g = StateGraph(MessagesState)

        g.add_node("logger", log_user_message)
        g.add_node("status_node", status_node)
        g.add_node("greet_node", greet_node)
        g.add_node("help_node", help_node)
        g.add_node("time_node", time_node)
        g.add_node("test_db_node", test_db_node)
        g.add_node("sample_spend_node", sample_spend_node)
        g.add_node("top_suppliers_node", top_suppliers_node)
        g.add_node("list_tables_node", list_tables_node)
        g.add_node("create_spend_table_node", create_spend_table_node)
        g.add_node("insert_test_data_node", insert_test_data_node)
        g.add_node("maverick_spend_node", maverick_spend_node)
        g.add_node("spend_leakage_node", spend_leakage_node)
        g.add_node("spend_by_category_node", spend_by_category_node)
        g.add_node("savings_opportunities_node", savings_opportunities_node)
        g.add_node("supplier_risk_node", supplier_risk_node)
        g.add_node("spend_trend_node", spend_trend_node)
        g.add_node("ask_node", ask_node)

        g.add_edge(START, "logger")

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
                "create_spend_table_node": "create_spend_table_node",
                "insert_test_data_node": "insert_test_data_node",
                "maverick_spend_node": "maverick_spend_node",
                "spend_leakage_node": "spend_leakage_node",
                "spend_by_category_node": "spend_by_category_node",
                "savings_opportunities_node": "savings_opportunities_node",
                "supplier_risk_node": "supplier_risk_node",
                "spend_trend_node": "spend_trend_node",
                "ask_node": "ask_node",
            },
        )

        g.add_edge("status_node", END)
        g.add_edge("greet_node", END)
        g.add_edge("help_node", END)
        g.add_edge("time_node", END)
        g.add_edge("test_db_node", END)
        g.add_edge("sample_spend_node", END)
        g.add_edge("top_suppliers_node", END)
        g.add_edge("list_tables_node", END)
        g.add_edge("create_spend_table_node", END)
        g.add_edge("insert_test_data_node", END)
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
