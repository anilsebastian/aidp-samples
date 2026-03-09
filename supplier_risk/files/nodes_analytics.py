# ============================================================
# nodes_analytics.py — All 7 Analytics Commands
# ============================================================
# Confirmed patterns from connectivity test:
#
# SQL: make_sql_tool(name, desc, query) → tool.ainvoke({})
#      result shape: result["result"]["structuredContent"]["rows"]
#      extract with: extract_rows_from_result(result)
#
# RAG: make_rag_tool(name, desc) → tool.ainvoke({"query": "..."})
#      result shape: result["result"]["structuredContent"]["answer"]
#      extract with: extract_rag_answer(result)
#
# Messages: {"role": "ai", "content": msg}  ← plain dict, not AIMessage

import traceback
from langgraph.graph import MessagesState
from config import CATALOG_KEY, SCHEMA_KEY, SUPPLIER_MASTER_TABLE, SPEND_HISTORY_TABLE, PO_PIPELINE_TABLE
from sql_tools import make_sql_tool
from rag_tools import make_rag_tool
from helpers import (
    extract_text_from_last_message, extract_rows_from_result, extract_rag_answer,
    format_currency, risk_color, delivery_color, make_bar
)


def _entity(state, strip_words):
    """Extract entity name from user message by stripping command keywords."""
    text = extract_text_from_last_message(state).lower().strip()
    for w in strip_words:
        text = text.replace(w, "").strip()
    return text.strip()


# ── Node 1: Supplier Risk Profile (SQL + RAG) ──────────────────────────────

async def supplier_risk_profile_node(state: MessagesState):
    entity = _entity(state, ["supplier risk profile", "risk profile"])
    if not entity:
        return {"messages": [{"role": "ai", "content": "Please specify a supplier name. Example: `supplier risk profile acme`"}]}
    try:
        master_q = f"""
            SELECT "supplier_id", "supplier_name", "country", "category",
                   "tier", "payment_terms", "active_since",
                   TO_NUMBER("risk_score_internal") AS risk_score
            FROM {SUPPLIER_MASTER_TABLE}
            WHERE LOWER("supplier_name") LIKE '%{entity}%'
            FETCH FIRST 1 ROWS ONLY
        """
        history_q = f"""
            SELECT h."quarter",
                   TO_NUMBER(h."spend_usd") AS spend_usd,
                   TO_NUMBER(h."on_time_delivery_pct") AS on_time_pct,
                   TO_NUMBER(h."defect_rate_pct") AS defect_rate,
                   TO_NUMBER(h."payment_delays") AS payment_delays
            FROM {SPEND_HISTORY_TABLE} h
            JOIN {SUPPLIER_MASTER_TABLE} m ON h."supplier_id" = m."supplier_id"
            WHERE LOWER(m."supplier_name") LIKE '%{entity}%'
            ORDER BY h."quarter" ASC
        """
        pipeline_q = f"""
            SELECT p."po_id", p."item_description",
                   TO_NUMBER(p."po_value_usd") AS po_value_usd,
                   p."expected_delivery", p."criticality", p."sourcing_alternative"
            FROM {PO_PIPELINE_TABLE} p
            JOIN {SUPPLIER_MASTER_TABLE} m ON p."supplier_id" = m."supplier_id"
            WHERE LOWER(m."supplier_name") LIKE '%{entity}%'
        """
        master_rows   = extract_rows_from_result(await make_sql_tool("profile_master",   "master",   master_q).ainvoke({}))
        history_rows  = extract_rows_from_result(await make_sql_tool("profile_history",  "history",  history_q).ainvoke({}))
        pipeline_rows = extract_rows_from_result(await make_sql_tool("profile_pipeline", "pipeline", pipeline_q).ainvoke({}))
        rag_answer    = extract_rag_answer(await make_rag_tool("profile_rag", "supplier risk news").ainvoke({"query": f"risk alerts news financial operational {entity}"}))

        if not master_rows:
            return {"messages": [{"role": "ai", "content": f"No supplier found matching '{entity}'."}]}

        m     = master_rows[0]
        risk  = m.get("RISK_SCORE", "N/A")
        name  = m.get("SUPPLIER_NAME", entity)

        lines = [
            f"## {risk_color(risk)} Supplier Risk Profile: {name}",
            f"",
            f"| Field | Value |", f"|---|---|",
            f"| Country | {m.get('COUNTRY','')} |",
            f"| Category | {m.get('CATEGORY','')} |",
            f"| Tier | {m.get('TIER','')} |",
            f"| Payment Terms | {m.get('PAYMENT_TERMS','')} |",
            f"| Active Since | {m.get('ACTIVE_SINCE','')} |",
            f"| **Risk Score** | **{risk}/100** {risk_color(risk)} |",
            f"",
            f"### 📊 Performance Trend",
            f"| Quarter | Spend | On-Time % | Defect % | Payment Delays |",
            f"|---|---|---|---|---|",
        ]
        for h in history_rows:
            ot = h.get("ON_TIME_PCT", 0)
            lines.append(f"| {h.get('QUARTER','')} | {format_currency(h.get('SPEND_USD',0))} | {delivery_color(ot)} {ot}% | {h.get('DEFECT_RATE',0)}% | {h.get('PAYMENT_DELAYS',0)} |")

        if pipeline_rows:
            lines += ["", "### 🚚 Open POs", "| PO ID | Description | Value | Delivery | Criticality | Backup |", "|---|---|---|---|---|---|"]
            for p in pipeline_rows:
                lines.append(f"| {p.get('PO_ID','')} | {p.get('ITEM_DESCRIPTION','')} | {format_currency(p.get('PO_VALUE_USD',0))} | {p.get('EXPECTED_DELIVERY','')} | {p.get('CRITICALITY','')} | {p.get('SOURCING_ALTERNATIVE','')} |")

        if rag_answer:
            lines += ["", "### 🧠 Risk Intelligence", rag_answer]

        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        return {"messages": [{"role": "ai", "content": f"❌ Error: {e}\n{traceback.format_exc()}"}]}


# ── Node 2: High Risk Suppliers (SQL + RAG) ────────────────────────────────

async def high_risk_suppliers_node(state: MessagesState):
    try:
        q = f"""
            SELECT m."supplier_name", m."country", m."category", m."tier",
                   TO_NUMBER(m."risk_score_internal") AS risk_score,
                   TO_NUMBER(h."spend_usd") AS latest_spend,
                   TO_NUMBER(h."on_time_delivery_pct") AS on_time_pct,
                   TO_NUMBER(h."defect_rate_pct") AS defect_rate
            FROM {SUPPLIER_MASTER_TABLE} m
            JOIN {SPEND_HISTORY_TABLE} h ON m."supplier_id" = h."supplier_id"
            WHERE TO_NUMBER(m."risk_score_internal") > 70
            AND h."quarter" = 'Q4-2024'
            ORDER BY TO_NUMBER(m."risk_score_internal") DESC
        """
        rows       = extract_rows_from_result(await make_sql_tool("high_risk_sql", "high risk", q).ainvoke({}))
        rag_answer = extract_rag_answer(await make_rag_tool("high_risk_rag", "high risk news").ainvoke({"query": "high risk supplier financial distress operational failure geopolitical compliance"}))

        if not rows:
            return {"messages": [{"role": "ai", "content": "No suppliers found with risk score above 70."}]}

        lines = ["## 🔴 High Risk Suppliers", "", "| Supplier | Country | Category | Risk Score | Q4 Spend | On-Time % | Defect % |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            score = r.get("RISK_SCORE", 0)
            lines.append(f"| {r.get('SUPPLIER_NAME','')} | {r.get('COUNTRY','')} | {r.get('CATEGORY','')} | {risk_color(score)} **{score}** | {format_currency(r.get('LATEST_SPEND',0))} | {delivery_color(r.get('ON_TIME_PCT',0))} {r.get('ON_TIME_PCT',0)}% | {r.get('DEFECT_RATE',0)}% |")

        if rag_answer:
            lines += ["", "### 🧠 Risk Intelligence Summary", rag_answer]

        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        return {"messages": [{"role": "ai", "content": f"❌ Error: {e}\n{traceback.format_exc()}"}]}


# ── Node 3: Open PO Exposure (SQL only) ───────────────────────────────────

async def open_po_exposure_node(state: MessagesState):
    try:
        q = f"""
            SELECT p."po_id", m."supplier_name", m."country",
                   TO_NUMBER(m."risk_score_internal") AS risk_score,
                   p."item_description",
                   TO_NUMBER(p."po_value_usd") AS po_value_usd,
                   p."expected_delivery", p."criticality", p."sourcing_alternative"
            FROM {PO_PIPELINE_TABLE} p
            JOIN {SUPPLIER_MASTER_TABLE} m ON p."supplier_id" = m."supplier_id"
            ORDER BY TO_NUMBER(m."risk_score_internal") DESC, TO_NUMBER(p."po_value_usd") DESC
        """
        rows = extract_rows_from_result(await make_sql_tool("open_po", "open po", q).ainvoke({}))
        if not rows:
            return {"messages": [{"role": "ai", "content": "No open POs found."}]}

        total      = sum(float(r.get("PO_VALUE_USD", 0) or 0) for r in rows)
        high_risk  = sum(float(r.get("PO_VALUE_USD", 0) or 0) for r in rows if int(r.get("RISK_SCORE", 0) or 0) > 70)
        pct        = (high_risk / total * 100) if total > 0 else 0

        lines = [
            "## 🚚 Open PO Exposure", "",
            f"**Total Pipeline:** {format_currency(total)} | **At-Risk (score >70):** {format_currency(high_risk)} ({pct:.0f}%)", "",
            "| PO ID | Supplier | Risk | Value | Delivery | Criticality | Backup Source |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            score = r.get("RISK_SCORE", 0)
            lines.append(f"| {r.get('PO_ID','')} | {r.get('SUPPLIER_NAME','')} | {risk_color(score)} {score} | {format_currency(r.get('PO_VALUE_USD',0))} | {r.get('EXPECTED_DELIVERY','')} | {r.get('CRITICALITY','')} | {r.get('SOURCING_ALTERNATIVE','')} |")

        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        return {"messages": [{"role": "ai", "content": f"❌ Error: {e}\n{traceback.format_exc()}"}]}


# ── Node 4: Delivery Performance (SQL only) ───────────────────────────────

async def delivery_performance_node(state: MessagesState):
    entity = _entity(state, ["delivery performance", "delivery trend", "delivery"])
    if not entity:
        return {"messages": [{"role": "ai", "content": "Please specify a supplier name. Example: `delivery performance acme`"}]}
    try:
        q = f"""
            SELECT m."supplier_name", h."quarter",
                   TO_NUMBER(h."spend_usd") AS spend_usd,
                   TO_NUMBER(h."po_count") AS po_count,
                   TO_NUMBER(h."on_time_delivery_pct") AS on_time_pct,
                   TO_NUMBER(h."defect_rate_pct") AS defect_rate,
                   TO_NUMBER(h."payment_delays") AS payment_delays
            FROM {SPEND_HISTORY_TABLE} h
            JOIN {SUPPLIER_MASTER_TABLE} m ON h."supplier_id" = m."supplier_id"
            WHERE LOWER(m."supplier_name") LIKE '%{entity}%'
            ORDER BY h."quarter" ASC
        """
        rows = extract_rows_from_result(await make_sql_tool("delivery_perf", "delivery", q).ainvoke({}))
        if not rows:
            return {"messages": [{"role": "ai", "content": f"No delivery data found for '{entity}'."}]}

        name  = rows[0].get("SUPPLIER_NAME", entity)
        lines = [f"## 📦 Delivery Performance: {name}", "", "| Quarter | Spend | POs | On-Time % | Defect % | Payment Delays |", "|---|---|---|---|---|---|"]
        prev_ot = None
        for r in rows:
            ot    = float(r.get("ON_TIME_PCT", 0) or 0)
            trend = ("📈" if ot > prev_ot else "📉" if ot < prev_ot else "➡️") if prev_ot is not None else ""
            prev_ot = ot
            lines.append(f"| {r.get('QUARTER','')} | {format_currency(r.get('SPEND_USD',0))} | {r.get('PO_COUNT',0)} | {delivery_color(ot)} {ot}% {trend} | {r.get('DEFECT_RATE',0)}% | {r.get('PAYMENT_DELAYS',0)} |")

        if len(rows) >= 2:
            delta = float(rows[-1].get("ON_TIME_PCT", 0) or 0) - float(rows[0].get("ON_TIME_PCT", 0) or 0)
            if delta < -10:
                lines.append(f"\n🔴 **Deteriorating: on-time delivery dropped {abs(delta):.0f} pts over the period.**")
            elif delta > 5:
                lines.append(f"\n🟢 **Improving: on-time delivery up {delta:.0f} pts over the period.**")
            else:
                lines.append(f"\n🟡 **Stable trend over the period.**")

        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        return {"messages": [{"role": "ai", "content": f"❌ Error: {e}\n{traceback.format_exc()}"}]}


# ── Node 5: Supplier News Alerts (RAG only) ───────────────────────────────

async def supplier_news_alerts_node(state: MessagesState):
    entity = _entity(state, ["supplier news alerts", "news alerts", "news", "alerts"])
    try:
        query      = f"risk alerts news financial operational geopolitical compliance {entity}" if entity else "supplier risk alerts news"
        rag_answer = extract_rag_answer(await make_rag_tool("news_alerts", "supplier news").ainvoke({"query": query}))
        title      = f"## 📰 Supplier News & Risk Alerts{f': {entity.title()}' if entity else ''}\n\n"
        return {"messages": [{"role": "ai", "content": title + (rag_answer or "No relevant alerts found.")}]}
    except Exception as e:
        return {"messages": [{"role": "ai", "content": f"❌ Error: {e}\n{traceback.format_exc()}"}]}


# ── Node 6: ESG Compliance (RAG only) ─────────────────────────────────────

async def esg_compliance_node(state: MessagesState):
    try:
        rag_answer = extract_rag_answer(await make_rag_tool("esg_rag", "esg compliance").ainvoke({"query": "ESG compliance sustainability audit certification supplier forced labor"}))
        return {"messages": [{"role": "ai", "content": "## 🌱 ESG & Compliance Status\n\n" + (rag_answer or "No ESG information found.")}]}
    except Exception as e:
        return {"messages": [{"role": "ai", "content": f"❌ Error: {e}\n{traceback.format_exc()}"}]}


# ── Node 7: Category Risk Briefing (SQL + RAG) ────────────────────────────

async def category_risk_briefing_node(state: MessagesState):
    entity = _entity(state, ["category risk briefing", "category risk", "category briefing", "category"])
    if not entity:
        return {"messages": [{"role": "ai", "content": "Please specify a category. Example: `category risk briefing electronics`"}]}
    try:
        q = f"""
            SELECT m."supplier_name", m."country", m."tier",
                   TO_NUMBER(m."risk_score_internal") AS risk_score,
                   SUM(TO_NUMBER(h."spend_usd")) AS total_spend
            FROM {SUPPLIER_MASTER_TABLE} m
            JOIN {SPEND_HISTORY_TABLE} h ON m."supplier_id" = h."supplier_id"
            WHERE LOWER(m."category") LIKE '%{entity.lower()}%'
            GROUP BY m."supplier_name", m."country", m."tier", m."risk_score_internal"
            ORDER BY TO_NUMBER(m."risk_score_internal") DESC
        """
        rows       = extract_rows_from_result(await make_sql_tool("cat_risk_sql", "category risk", q).ainvoke({}))
        rag_answer = extract_rag_answer(await make_rag_tool("cat_risk_rag", "category risk news").ainvoke({"query": f"{entity} category supply chain risk disruption tariff geopolitical"}))

        if not rows:
            return {"messages": [{"role": "ai", "content": f"No supplier data found for category '{entity}'."}]}

        total_spend = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows)
        lines = [
            f"## 🏷️ Category Risk Briefing: {entity.title()}", "",
            f"**Total Category Spend:** {format_currency(total_spend)}", "",
            "| Supplier | Country | Tier | Risk Score | Total Spend |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            score = r.get("RISK_SCORE", 0)
            lines.append(f"| {r.get('SUPPLIER_NAME','')} | {r.get('COUNTRY','')} | {r.get('TIER','')} | {risk_color(score)} {score} | {format_currency(r.get('TOTAL_SPEND',0))} |")

        if rag_answer:
            lines += ["", "### 🧠 Category Risk Assessment", rag_answer]

        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        return {"messages": [{"role": "ai", "content": f"❌ Error: {e}\n{traceback.format_exc()}"}]}
