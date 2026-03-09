# ============================================================
# nodes_nlp.py — Natural language intent understanding node
# ============================================================
import traceback
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState

from llm import LLM_AVAILABLE, ensure_llm, SPEND_INTENT_PROMPT, parse_intent_response
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
            prefix = f"💡 *{explanation}*\n\n" if explanation else ""
            result = await intent_map[intent](state)
            
            if result and "messages" in result and result["messages"]:
                original_content = result["messages"][0].get("content", "")
                result["messages"][0]["content"] = prefix + original_content
            
            return result
        else:
            return await help_node(state)
            
    except Exception as e:
        tb = traceback.format_exc()
        print("ask_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"I had trouble understanding that. Try specific commands like `top suppliers` or `maverick spend`.\n\nError: {e}"}]}
