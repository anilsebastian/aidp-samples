# ============================================================
# rag_tools.py — RAG Tool Factory
# ============================================================
from typing import List
from config import RAG_CATALOG, RAG_SCHEMA, RAG_KNOWLEDGE_BASE, RAG_TOP_K, REGION, MODEL_ID, MODEL_PROVIDER, COMPARTMENT_ID

IMPORT_ERRORS: List[str] = []
RAG_AVAILABLE = True

try:
    from aidputils.agents.toolkit.configs import AIDPToolConf
    from aidputils.agents.toolkit.tool_helper import create_langgraph_tool
except Exception as e:
    RAG_AVAILABLE = False
    IMPORT_ERRORS.append(f"aidputils import failed: {e!r}")

LLM_CFG = {
    "model_id":       MODEL_ID,
    "model_provider": MODEL_PROVIDER,
    "compartment_id": COMPARTMENT_ID,
    "endpoint":       f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
}


def make_rag_tool(name: str, description: str):
    """
    Create a RAGTool pointing to supplier_risk_kb.
    Confirmed working pattern from connectivity test:
    - conf uses catalog / schema / knowledgeBase / topK / llm
    - params requires 'query' field (no defaultValue)
    - invoke with ainvoke({"query": "..."})
    """
    if not RAG_AVAILABLE:
        raise RuntimeError("RAGTool unavailable (aidputils import failed).")

    conf = AIDPToolConf(
        name=name,
        description=description,
        tool_class="RAGTool",
        conf={
            "catalog":       RAG_CATALOG,
            "schema":        RAG_SCHEMA,
            "knowledgeBase": RAG_KNOWLEDGE_BASE,
            "topK":          RAG_TOP_K,
            "llm":           LLM_CFG,
        },
        params=[
            {
                "name":        "query",
                "type":        "string",
                "description": "search query for knowledge base",
            }
        ],
    )
    return create_langgraph_tool(conf.model_dump())
