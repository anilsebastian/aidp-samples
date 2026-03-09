# ============================================================
# config.py — AIDP Spend Agent Configuration
# ============================================================
import os

# Database Configuration
CATALOG_KEY = os.getenv("CATALOG_KEY", "custom_fdibundletest")
SCHEMA_KEY = os.getenv("SCHEMA_KEY", "fdi_aidp_cust01")
SPEND_TABLE = os.getenv("SPEND_TABLE", "HARMONIZED_SPEND")

# LLM Configuration (OCI GenAI)
REGION = os.getenv("REGION", "us-phoenix-1")
MODEL_ID = os.getenv("MODEL_ID", "xai.grok-3-mini")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "generic")
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..aaaaaaaaoyio62q3gtybcicjloxahujztqz5tn4dgzsrtxybx2smbdv4vhva")

# LLM Parameters
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024
