# ============================================================
# nodes_analytics.py — Analytics nodes
# ============================================================
import traceback
from langgraph.graph import MessagesState

from config import SPEND_TABLE
from helpers import (
    format_currency, make_bar, risk_emoji,
    extract_rows_from_result, format_sqltool_result
)
from sql_tools import make_sql_tool


async def top_suppliers_node(state: MessagesState):
    """Aggregate spend by canonical supplier with visual bar chart."""
    try:
        sql = f'''SELECT CANONICAL_SUPPLIER_NAME AS supplier, SUM(LINE_AMOUNT_USD) AS spend_usd
FROM {SPEND_TABLE}
WHERE CANONICAL_SUPPLIER_NAME IS NOT NULL
GROUP BY CANONICAL_SUPPLIER_NAME
ORDER BY spend_usd DESC
FETCH FIRST 10 ROWS ONLY'''

        print(f"DEBUG executing SQL: {sql}")
        tool = make_sql_tool(
            name="top_suppliers",
            description="Top suppliers by spend",
            query=sql,
        )
        result = await tool.ainvoke({})
        rows = extract_rows_from_result(result)
        
        if not rows:
            return {"messages": [{"role": "ai", "content": "No supplier data found. Try `sample spend` to verify data exists."}]}
        
        total_spend = sum(float(r.get("SPEND_USD", 0) or 0) for r in rows)
        max_spend = max(float(r.get("SPEND_USD", 0) or 0) for r in rows) if rows else 0
        
        lines = []
        lines.append("📊 **Top 10 Suppliers by Spend**\n")
        lines.append(f"Total across top 10: {format_currency(total_spend)}\n")
        lines.append("| Rank | Supplier | Spend | |")
        lines.append("| --- | --- | --- | --- |")
        
        for i, row in enumerate(rows, 1):
            supplier = row.get("SUPPLIER", row.get("supplier", "Unknown"))
            spend = float(row.get("SPEND_USD", row.get("spend_usd", 0)) or 0)
            bar = make_bar(spend, max_spend, 10, "blue")
            lines.append(f"| {i} | {supplier} | {format_currency(spend)} | {bar} |")
        
        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("top_suppliers_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"Top suppliers query failed: {e}\n\nTRACE:\n{tb}"}]}


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
        tool = make_sql_tool(
            name="maverick_spend",
            description="Detect maverick spend",
            query=sql,
        )
        result = await tool.ainvoke({})
        rows = extract_rows_from_result(result)
        
        if not rows:
            return {"messages": [{"role": "ai", "content": "✅ No maverick spend detected. All transactions have contracts and matched suppliers."}]}
        
        total_maverick = sum(float(r.get("LINE_AMOUNT_USD", 0) or 0) for r in rows)
        no_contract_count = sum(1 for r in rows if "No Contract" in str(r.get("RISK_TYPE", "")))
        unmatched_count = sum(1 for r in rows if "Unmatched" in str(r.get("RISK_TYPE", "")))
        low_conf_count = sum(1 for r in rows if "Low Confidence" in str(r.get("RISK_TYPE", "")))
        
        lines = []
        lines.append("🚨 **Maverick Spend Analysis**\n")
        lines.append(f"**Total Flagged:** {format_currency(total_maverick)} across {len(rows)} transactions\n")
        lines.append("**Breakdown:**")
        if no_contract_count > 0:
            lines.append(f"  🟠 No Contract: {no_contract_count} transactions")
        if unmatched_count > 0:
            lines.append(f"  🟠 Unmatched Supplier: {unmatched_count} transactions")
        if low_conf_count > 0:
            lines.append(f"  🟡 Low Confidence Match: {low_conf_count} transactions")
        lines.append("")
        
        lines.append("| Risk | Supplier | Category | Amount |")
        lines.append("| --- | --- | --- | --- |")
        
        for row in rows[:15]:
            risk = row.get("RISK_TYPE", "")
            emoji = risk_emoji(risk)
            supplier = row.get("SUPPLIER_NAME", "Unknown")[:25]
            category = row.get("CATEGORY_NAME", "")[:20]
            amount = float(row.get("LINE_AMOUNT_USD", 0) or 0)
            lines.append(f"| {emoji} | {supplier} | {category} | {format_currency(amount)} |")
        
        if len(rows) > 15:
            lines.append(f"\n*Showing top 15 of {len(rows)} flagged transactions*")
        
        lines.append("\n**Recommendation:** Review no-contract purchases for compliance and negotiate master agreements with frequently used suppliers.")
        
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
        tool = make_sql_tool(
            name="spend_leakage",
            description="Identify spend consolidation opportunities",
            query=sql,
        )
        result = await tool.ainvoke({})
        rows = extract_rows_from_result(result)
        
        if not rows:
            return {"messages": [{"role": "ai", "content": "✅ No spend leakage detected. Each supplier is being managed through a single system."}]}
        
        total_leakage = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows)
        max_spend = max(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows) if rows else 0
        
        lines = []
        lines.append("💰 **Spend Consolidation Opportunities**\n")
        lines.append(f"**Total Addressable Spend:** {format_currency(total_leakage)} across {len(rows)} suppliers\n")
        lines.append("Suppliers being used across multiple ERP systems — opportunity for contract consolidation:\n")
        
        lines.append("| Supplier | Systems | Spend | |")
        lines.append("| --- | --- | --- | --- |")
        
        for row in rows:
            supplier = str(row.get("CANONICAL_SUPPLIER_NAME", "Unknown"))[:25]
            num_sys = row.get("NUM_SYSTEMS", 0)
            spend = float(row.get("TOTAL_SPEND", 0) or 0)
            bar = make_bar(spend, max_spend, 8, "green")
            
            sys_indicator = "🔴" if num_sys >= 3 else "🟡"
            lines.append(f"| {supplier} | {sys_indicator} {num_sys} | {format_currency(spend)} | {bar} |")
        
        lines.append("")
        lines.append("**Legend:** 🔴 3+ systems (high priority) | 🟡 2 systems")
        lines.append("")
        lines.append("**Recommendation:** Consolidate these suppliers under single enterprise contracts to improve negotiating leverage and reduce procurement overhead.")
        
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
        tool = make_sql_tool(
            name="spend_by_category",
            description="Analyze spend by category",
            query=sql,
        )
        result = await tool.ainvoke({})
        rows = extract_rows_from_result(result)
        
        if not rows:
            return {"messages": [{"role": "ai", "content": "No category data found."}]}
        
        total_spend = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows)
        max_spend = max(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows) if rows else 0
        high_risk_categories = [r for r in rows if float(r.get("PCT_OFF_CONTRACT", 0) or 0) > 50]
        
        lines = []
        lines.append("📊 **Spend by Category**\n")
        lines.append(f"**Total Spend:** {format_currency(total_spend)} across {len(rows)} categories")
        if high_risk_categories:
            lines.append(f"**⚠️ High Risk:** {len(high_risk_categories)} categories with >50% off-contract spend\n")
        else:
            lines.append("")
        
        lines.append("| Category | Spend | | Off-Contract | Risk |")
        lines.append("| --- | --- | --- | --- | --- |")
        
        for row in rows:
            category = str(row.get("CATEGORY_NAME", "Unknown"))[:25]
            spend = float(row.get("TOTAL_SPEND", 0) or 0)
            bar = make_bar(spend, max_spend, 8, "purple")
            pct_off = float(row.get("PCT_OFF_CONTRACT", 0) or 0)
            
            if pct_off >= 75:
                risk = "🔴 High"
            elif pct_off >= 50:
                risk = "🟠 Medium"
            elif pct_off >= 25:
                risk = "🟡 Low"
            else:
                risk = "🟢 OK"
            
            lines.append(f"| {category} | {format_currency(spend)} | {bar} | {pct_off:.0f}% | {risk} |")
        
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
        tool = make_sql_tool(
            name="savings_opportunities",
            description="Identify price variance savings opportunities",
            query=sql,
        )
        result = await tool.ainvoke({})
        rows = extract_rows_from_result(result)
        
        if not rows:
            return {"messages": [{"role": "ai", "content": "✅ No significant price variances detected. Pricing appears consistent across business units."}]}
        
        total_addressable = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows)
        est_savings = total_addressable * 0.05
        
        lines = []
        lines.append("💵 **Savings Opportunities — Price Variance Analysis**\n")
        lines.append(f"**Addressable Spend:** {format_currency(total_addressable)}")
        lines.append(f"**Estimated Savings Potential:** {format_currency(est_savings)} (5% of addressable)\n")
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
            
            lines.append(f"| {supplier} | {category} | {num_bus} | {format_currency(min_price)} | {format_currency(max_price)} | {var_display} |")
        
        lines.append("")
        lines.append("**Recommendation:** Negotiate standardized pricing with high-variance suppliers. Start with largest variance × volume combinations.")
        
        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("savings_opportunities_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"Savings analysis failed: {e}\n\nTRACE:\n{tb}"}]}


async def supplier_risk_node(state: MessagesState):
    """Analyze supplier concentration risk - single source categories, over-reliance."""
    try:
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
        
        tool1 = make_sql_tool(
            name="supplier_concentration",
            description="Single supplier categories",
            query=sql_concentration,
        )
        result1 = await tool1.ainvoke({})
        concentration_rows = extract_rows_from_result(result1)
        
        tool2 = make_sql_tool(
            name="supplier_reliance",
            description="Supplier over-reliance",
            query=sql_reliance,
        )
        result2 = await tool2.ainvoke({})
        reliance_rows = extract_rows_from_result(result2)
        
        lines = []
        lines.append("⚠️ **Supplier Risk Analysis**\n")
        
        if concentration_rows:
            single_source_spend = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in concentration_rows)
            lines.append(f"**🔴 Single-Source Categories:** {len(concentration_rows)} categories ({format_currency(single_source_spend)} at risk)\n")
            lines.append("| Category | Sole Supplier | Spend |")
            lines.append("| --- | --- | --- |")
            for row in concentration_rows[:6]:
                cat = str(row.get("CATEGORY_NAME", ""))[:22]
                supplier = str(row.get("PRIMARY_SUPPLIER", ""))[:22]
                spend = float(row.get("TOTAL_SPEND", 0) or 0)
                lines.append(f"| {cat} | {supplier} | {format_currency(spend)} |")
            lines.append("")
        else:
            lines.append("✅ **Single-Source:** No high-value single-source categories detected.\n")
        
        if reliance_rows:
            lines.append("**🟠 Supplier Concentration:**\n")
            lines.append("| Supplier | Spend | % of Total | Categories |")
            lines.append("| --- | --- | --- | --- |")
            max_spend = max(float(r.get("SUPPLIER_SPEND", 0) or 0) for r in reliance_rows) if reliance_rows else 0
            for row in reliance_rows[:8]:
                supplier = str(row.get("CANONICAL_SUPPLIER_NAME", ""))[:22]
                spend = float(row.get("SUPPLIER_SPEND", 0) or 0)
                pct = float(row.get("PCT_OF_TOTAL", 0) or 0)
                cats = row.get("CATEGORIES_SERVED", 0)
                bar = make_bar(spend, max_spend, 6, "orange")
                lines.append(f"| {supplier} | {format_currency(spend)} | {pct:.1f}% {bar} | {cats} |")
            lines.append("")
        else:
            lines.append("✅ **Concentration:** No suppliers exceed 5% of total spend.\n")
        
        lines.append("**Recommendations:**")
        if concentration_rows:
            lines.append("• Develop alternate suppliers for single-source categories")
        if reliance_rows:
            lines.append("• Review contracts for top suppliers — ensure favorable terms given volume")
        lines.append("• Consider strategic sourcing initiative for high-concentration areas")
        
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
        tool = make_sql_tool(
            name="spend_trend",
            description="Analyze spend trends over time",
            query=sql,
        )
        result = await tool.ainvoke({})
        rows = extract_rows_from_result(result)
        
        if not rows:
            return {"messages": [{"role": "ai", "content": "No trend data available. Check that PO_DATE is populated."}]}
        
        rows = list(reversed(rows))
        
        total_spend = sum(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows)
        avg_monthly = total_spend / len(rows) if rows else 0
        max_spend = max(float(r.get("TOTAL_SPEND", 0) or 0) for r in rows) if rows else 0
        
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
        lines.append(f"**Total:** {format_currency(total_spend)} | **Avg Monthly:** {format_currency(avg_monthly)} | **Trend:** {trend}\n")
        
        lines.append("| Month | Spend | | Maverick % |")
        lines.append("| --- | --- | --- | --- |")
        
        for row in rows:
            month = row.get("MONTH", "")
            spend = float(row.get("TOTAL_SPEND", 0) or 0)
            bar = make_bar(spend, max_spend, 10, "blue")
            maverick_pct = float(row.get("PCT_MAVERICK", 0) or 0)
            
            mav_indicator = "🔴" if maverick_pct > 50 else "🟡" if maverick_pct > 25 else "🟢"
            lines.append(f"| {month} | {format_currency(spend)} | {bar} | {mav_indicator} {maverick_pct:.0f}% |")
        
        lines.append("")
        lines.append("**Legend:** Maverick % = spend without contracts")
        
        return {"messages": [{"role": "ai", "content": "\n".join(lines)}]}
    except Exception as e:
        tb = traceback.format_exc()
        print("spend_trend_node error:", e)
        print(tb)
        return {"messages": [{"role": "ai", "content": f"Spend trend analysis failed: {e}\n\nTRACE:\n{tb}"}]}
