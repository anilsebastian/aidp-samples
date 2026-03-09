# ============================================================
# llm.py — LLM initialization
# ============================================================
from typing import List
from config import REGION, MODEL_ID, MODEL_PROVIDER, COMPARTMENT_ID, LLM_TEMPERATURE, LLM_MAX_TOKENS

IMPORT_ERRORS: List[str] = []
LLM_AVAILABLE = True

try:
    from aidputils.agents.toolkit.agent_helper import init_oci_llm, pre_invoke_setup
    from aidputils.agents.toolkit.configs import OCIAIConf
except Exception as e:
    LLM_AVAILABLE = False
    IMPORT_ERRORS.append(f"LLM imports failed: {e!r}")

_llm_instance = None


def ensure_llm():
    """Lazy-init the LLM. Call pre_invoke_setup() first (SpendAgent pattern)."""
    global _llm_instance
    if not LLM_AVAILABLE:
        raise RuntimeError("LLM not available.")
    pre_invoke_setup()
    if _llm_instance is None:
        oci_conf = OCIAIConf(
            model_provider=MODEL_PROVIDER,
            compartment_id=COMPARTMENT_ID,
            model_id=MODEL_ID,
            endpoint=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com",
            model_args={
                "temperature": LLM_TEMPERATURE,
                "max_tokens":  LLM_MAX_TOKENS,
            }
        )
        _llm_instance = init_oci_llm(oci_conf)
        print("LLM initialized")
    return _llm_instance
