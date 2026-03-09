# ============================================================
# sql_tools.py — SQL Tool Factory
# ============================================================
from typing import List
from config import CATALOG_KEY, SCHEMA_KEY

IMPORT_ERRORS: List[str] = []
AIDP_AVAILABLE = True

try:
    from aidputils.agents.toolkit.configs import AIDPToolConf
    from aidputils.agents.toolkit.tool_helper import create_langgraph_tool
except Exception as e:
    AIDP_AVAILABLE = False
    IMPORT_ERRORS.append(f"aidputils import failed: {e!r}")


def make_sql_tool(name: str, description: str, query: str):
    """
    Create a SQLTool for the given query.
    Confirmed working pattern from connectivity test.
    - conf uses catalogKey / schemaKey / query
    - params=[]
    - invoke with ainvoke({})
    """
    if not AIDP_AVAILABLE:
        raise RuntimeError("SQLTool unavailable (aidputils import failed).")

    conf = AIDPToolConf(
        name=name,
        description=description,
        tool_class="SQLTool",
        conf={
            "catalogKey": CATALOG_KEY,
            "schemaKey":  SCHEMA_KEY,
            "query":      query,
        },
        params=[],
    )
    return create_langgraph_tool(conf.model_dump())
