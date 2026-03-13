# ============================================================
# nodes_nlp.py — Natural language intent understanding node
# ============================================================
import traceback
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState

from llm import (
    LLM_AVAILABLE, 
    ensure_llm, 
    SPEND_INTENT_PROMPT, 
    TEXT_TO_SQL_PROMPT,
    parse_intent_response,
    clean_sql_response
)
from nodes_core import extract_text_from_last_message, help_node
from nodes_analytics import (
    top_suppliers_node,
    maverick_spend_node,
    spend_leakage_node,
    spend_by_category_node,
    savings_opportunities_node,
    supplier_risk_node,
    spend_trend_node,
)
from nodes_core import sample_spend_node
from sql_tools import make_sql_tool
from helpers import extract_rows_from_result, format_currency


# ============================================================
# Data Question Node (Text-to-SQL)
# ============================================================
async def data_question_node(state: MessagesState):
    """
    Handle arbitrary data questions via Text-to-SQL.
    
    Flow:
    1. Take user's natural language question
    2. Send to LLM with schema context to generate SQL
    3. Execute the SQL
    4. Format and return results
    """
    user_text = extract_text_from_last_message(state)
    generated_sql = None  # Track for error reporting
    
    try:
        if not LLM_AVAILABLE:
            return {"messages": [{"role": "ai", "content": "LLM not available for Text-to-SQL. Please use specific commands like `top suppliers`. Type `help` for options."}]}
        
        # Step 1: Generate SQL from natural language
        llm = ensure_llm()
        sql_chain = TEXT_TO_SQL_PROMPT | llm | StrOutputParser()
        
        print(f"DEBUG: Generating SQL for: {user_text}")
        raw_sql = await sql_chain.ainvoke({"user_question": user_text})
        generated_sql = clean_sql_response(raw_sql)
        print(f"DEBUG: Generated SQL: {generated_sql}")
        
        # Basic validation — must look like a SELECT
        if not generated_sql.upper().strip().startswith("SELECT"):
            return {"messages": [{"role": "ai", "content": f"I generated an invalid query. Please try rephrasing your question.\n\nGenerated: {generated_sql}"}]}
        
        # Step 2: Execute the SQL
        tool = make_sql_tool(
            name="data_question",
            description="Execute user's data question",
            query=generated_sql,
        )
        result = await tool.ainvoke({})
        rows = extract_rows_from_result(result)
        
        # Step 3: Format results
        lines = []
        lines.append(f"📊 **Results for:** _{user_text}_\n")
        
        if not rows:
            lines.append("No results found for your query.")
            lines.append(f"\n<details><summary>SQL executed</summary>\n\n```sql\n{generated_sql}\n```\n</details>")
            return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
        
        # Determine if this is an aggregation (few rows) or a list (many rows)
        if len(rows) <= 10:
            # Show as a formatted summary
            for row in rows:
                row_parts = []
                for key, val in row.items():
                    # Format currency-like values
                    if key.lower() in ("total_spend", "spend", "amount", "monthly_spend", "supplier_spend", "line_amount_usd", "q3_spend", "q1_spend", "q2_spend", "q4_spend"):
                        try:
                            row_parts.append(f"**{key}**: {format_currency(float(val))}")
                        except (ValueError, TypeError):
                            row_parts.append(f"**{key}**: {val}")
                    else:
                        row_parts.append(f"**{key}**: {val}")
                lines.append(" | ".join(row_parts))
        else:
            # Show as a table for longer results
            if rows:
                # Header
                headers = list(rows[0].keys())
                lines.append(" | ".join(headers))
                lines.append(" | ".join(["---"] * len(headers)))
                # Rows (limit to 20)
                for row in rows[:20]:
                    vals = []
                    for h in headers:
                        v = row.get(h, "")
                        # Format currency
                        if h.lower() in ("total_spend", "spend", "amount", "monthly_spend", "supplier_spend"):
                            try:
                                vals.append(format_currency(float(v)))
                            except (ValueError, TypeError):
                                vals.append(str(v))
                        else:
                            vals.append(str(v))
                    lines.append(" | ".join(vals))
                if len(rows) > 20:
                    lines.append(f"\n*... and {len(rows) - 20} more rows*")
        
        # Show the SQL for transparency
        lines.append(f"\n<details><summary>SQL executed</summary>\n\n```sql\n{generated_sql}\n```\n</details>")
        
        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
        
    except Exception as e:
        tb = traceback.format_exc()
        print("data_question_node error:", e)
        print(tb)
        
        # Provide helpful error message
        error_msg = str(e)
        if "ORA-" in error_msg:
            # Oracle error — likely SQL syntax issue
            return {"messages": [{"role": "ai", "content": f"The generated SQL had an error. Try rephrasing your question.\n\nError: {error_msg}\n\nGenerated SQL:\n```sql\n{generated_sql if generated_sql else 'N/A'}\n```"}]}
        else:
            return {"messages": [{"role": "ai", "content": f"I had trouble answering that question.\n\nError: {e}"}]}


# ============================================================
# Ask Node (Intent Router → calls appropriate node)
# ============================================================
async def ask_node(state: MessagesState):
    """Natural language intent understanding - routes to appropriate analysis."""
    user_text = extract_text_from_last_message(state)
    
    try:
        if not LLM_AVAILABLE:
            return {"messages": [{"role": "ai", "content": "LLM not available. Please use specific commands like `top suppliers`, `maverick spend`, etc. Type `help` for options."}]}
        
        llm = ensure_llm()
        chain = SPEND_INTENT_PROMPT | llm | StrOutputParser()
        
        print(f"DEBUG: Asking LLM to interpret: {user_text}")
        raw_response = await chain.ainvoke({"user_message": user_text})
        print(f"DEBUG: LLM response: {raw_response}")
        
        parsed = parse_intent_response(raw_response)
        intent = parsed.get("intent", "help")
        explanation = parsed.get("explanation", "")
        
        # Map intent to node function — now includes data_question
        intent_map = {
            "top_suppliers": top_suppliers_node,
            "maverick_spend": maverick_spend_node,
            "spend_leakage": spend_leakage_node,
            "spend_by_category": spend_by_category_node,
            "savings_opportunities": savings_opportunities_node,
            "supplier_risk": supplier_risk_node,
            "spend_trend": spend_trend_node,
            "sample_data": sample_spend_node,
            "data_question": data_question_node,  # Text-to-SQL fallback
            "help": help_node,
        }
        
        if intent in intent_map:
            prefix = f"💡 *{explanation}*\n\n" if explanation else ""
            result = await intent_map[intent](state)
            
            if result and "messages" in result and result["messages"]:
                original_content = result["messages"][0].get("content", "")
                result["messages"][0]["content"] = prefix + original_content
            
            return result
        else:
            # Unknown intent — try data_question as fallback
            print(f"DEBUG: Unknown intent '{intent}', falling back to data_question")
            return await data_question_node(state)
            
    except Exception as e:
        tb = traceback.format_exc()
        print("ask_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"I had trouble understanding that. Try specific commands like `top suppliers` or `maverick spend`.\n\nError: {e}"}]}