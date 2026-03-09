# ============================================================
# llm.py — LLM initialization and intent understanding
# ============================================================
import json
import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    REGION, MODEL_ID, MODEL_PROVIDER, OCI_COMPARTMENT_ID,
    LLM_TEMPERATURE, LLM_MAX_TOKENS
)

# ============================================================
# Import tracking
# ============================================================
IMPORT_ERRORS: List[str] = []
LLM_AVAILABLE = True

try:
    from aidputils.agents.toolkit.agent_helper import init_oci_llm, pre_invoke_setup
    from aidputils.agents.toolkit.configs import OCIAIConf
except Exception as e:
    LLM_AVAILABLE = False
    IMPORT_ERRORS.append(f"LLM imports failed: {e!r}")

# Module-level LLM instance (lazy init)
_llm_instance = None


def ensure_llm():
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
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
            }
        )
        _llm_instance = init_oci_llm(oci_conf)
        print("LLM initialized")
    return _llm_instance


# ============================================================
# Intent Understanding Prompt
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


def parse_intent_response(raw: str) -> dict:
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
