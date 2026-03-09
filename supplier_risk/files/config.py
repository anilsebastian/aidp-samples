# ============================================================
# config.py — SupplierRiskAgent Configuration
# ============================================================
import os

# SQL — External Catalog
CATALOG_KEY   = os.getenv("CATALOG_KEY",       "custom_fdibundletest")
SCHEMA_KEY    = os.getenv("SCHEMA_KEY",         "fdi_aidp_cust01")

SUPPLIER_MASTER_TABLE  = "SUPPLIER_MASTER"
SPEND_HISTORY_TABLE    = "SUPPLIER_SPEND_HISTORY"
PO_PIPELINE_TABLE      = "SUPPLIER_PO_PIPELINE"

# RAG — Knowledge Base
RAG_CATALOG        = "supplier_risk"
RAG_SCHEMA         = "silver"
RAG_KNOWLEDGE_BASE = "supplier_risk_kb"
RAG_TOP_K          = 3

# LLM
REGION         = os.getenv("REGION",             "us-phoenix-1")
MODEL_ID       = os.getenv("MODEL_ID",           "xai.grok-3-mini")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER",     "generic")
COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID", "ocid1.compartment.oc1..aaaaaaaaoyio62q3gtybcicjloxahujztqz5tn4dgzsrtxybx2smbdv4vhva")

LLM_MAX_TOKENS  = 1024
LLM_TEMPERATURE = 0.2
