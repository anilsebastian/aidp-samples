# ============================================================
# nodes_analytics.py — Analytics nodes
# FIX: All column names double-quoted to match External catalog
#      lowercase identifiers. Oracle treats unquoted names as
#      UPPERCASE, but External catalog stores them lowercase.
#      Also: transaction_date → po_date, document_number → po_id,
#      match_confidence < 90 → < 0.9
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
        sql = f'''SELECT "canonical_supplier_name" AS supplier, SUM(TO_NUMBER("line_amount_usd")) AS spend_usd
FROM {SPEND_TABLE}
WHERE "canonical_supplier_name" IS NOT NULL
GROUP BY "canonical_supplier_name"
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
        
        total_spend = sum(float(r.get("SPEND_USD", r.get("spend_usd", 0)) or 0) for r in rows)
        max_spend = max(float(r.get("SPEND_USD", r.get("spend_usd", 0)) or 0) for r in rows) if rows else 0
        
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
    "po_id",
    "supplier_name",
    "category_name",
    "business_unit",
    "line_amount_usd",
    "contract_reference",
    "match_method",
    "match_confidence",
    CASE 
        WHEN "contract_reference" IS NULL AND "match_method" = 'NO_MATCH' THEN 'No Contract + Unmatched Supplier'
        WHEN "contract_reference" IS NULL THEN 'No Contract'
        WHEN "match_method" = 'NO_MATCH' THEN 'Unmatched Supplier'
        WHEN TO_NUMBER("match_confidence") < 0.9 THEN 'Low Confidence Match'
        ELSE 'Other'
    END AS risk_type
FROM {SPEND_TABLE}
WHERE "contract_reference" IS NULL 
   OR "match_method" = 'NO_MATCH'
   OR TO_NUMBER("match_confidence") < 0.9
ORDER BY TO_NUMBER("line_amount_usd") DESC
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
        
        # Calculate summary stats
        total_maverick = sum(float(r.get("LINE_AMOUNT_USD", r.get("line_amount_usd", 0)) or 0) for r in rows)
        no_contract_rows = [r for r in rows if "No Contract" in str(r.get("RISK_TYPE", r.get("risk_type", "")))]
        no_contract_total = sum(float(r.get("LINE_AMOUNT_USD", r.get("line_amount_usd", 0)) or 0) for r in no_contract_rows)
        unmatched_count = sum(1 for r in rows if "Unmatched" in str(r.get("RISK_TYPE", r.get("risk_type", ""))))
        low_conf_count = sum(1 for r in rows if "Low Confidence" in str(r.get("RISK_TYPE", r.get("risk_type", ""))))
        
        # Find top offender
        top_supplier = rows[0].get("SUPPLIER_NAME", rows[0].get("supplier_name", "Unknown")) if rows else "Unknown"
        top_amount = float(rows[0].get("LINE_AMOUNT_USD", rows[0].get("line_amount_usd", 0)) or 0) if rows else 0
        
        # Find repeat offenders
        supplier_counts = {}
        supplier_totals = {}
        for r in rows:
            sup = r.get("SUPPLIER_NAME", r.get("supplier_name", "Unknown"))
            amt = float(r.get("LINE_AMOUNT_USD", r.get("line_amount_usd", 0)) or 0)
            supplier_counts[sup] = supplier_counts.get(sup, 0) + 1
            supplier_totals[sup] = supplier_totals.get(sup, 0) + amt
        repeat_offenders = [(s, supplier_counts[s], supplier_totals[s]) for s in supplier_counts if supplier_counts[s] > 1]
        repeat_offenders.sort(key=lambda x: x[2], reverse=True)
        
        lines = []
        lines.append("🚨 **Maverick Spend Analysis**\n")
        lines.append(f"**Total Flagged:** {format_currency(total_maverick)} across {len(rows)} transactions\n")
        lines.append("**Breakdown:**")
        if no_contract_rows:
            lines.append(f"  🟠 No Contract: {len(no_contract_rows)} transactions ({format_currency(no_contract_total)})")
        if unmatched_count > 0:
            lines.append(f"  🟠 Unmatched Supplier: {unmatched_count} transactions")
        if low_conf_count > 0:
            lines.append(f"  🟡 Low Confidence Match: {low_conf_count} transactions")
        lines.append("")
        
        lines.append("| Risk | Supplier | Category | Amount |")
        lines.append("| --- | --- | --- | --- |")
        
        for row in rows[:15]:
            risk = row.get("RISK_TYPE", row.get("risk_type", ""))
            emoji = risk_emoji(risk)
            supplier = str(row.get("SUPPLIER_NAME", row.get("supplier_name", "Unknown")))[:25]
            category = str(row.get("CATEGORY_NAME", row.get("category_name", "")))[:20]
            amount = float(row.get("LINE_AMOUNT_USD", row.get("line_amount_usd", 0)) or 0)
            lines.append(f"| {emoji} | {supplier} | {category} | {format_currency(amount)} |")
        
        if len(rows) > 15:
            lines.append(f"\n*Showing top 15 of {len(rows)} flagged transactions*")
        
        # Smart recommendations
        lines.append("\n---\n**🎯 Recommended Actions:**\n")
        lines.append(f"**1. Immediate:** Review {top_supplier} ({format_currency(top_amount)}) — your largest flagged purchase. Determine if this should be under a master agreement.")
        
        if repeat_offenders:
            top_repeat = repeat_offenders[0]
            lines.append(f"\n**2. Quick Win:** {top_repeat[0]} has {top_repeat[1]} flagged transactions totaling {format_currency(top_repeat[2])}. Negotiate a blanket contract.")
        
        if no_contract_total > 100000:
            lines.append(f"\n**3. Process Fix:** {format_currency(no_contract_total)} in no-contract spend. Consider requiring contract reference for POs over $10K.")
        
        lines.append(f"\n**Next:** Run `supplier risk` to check if these suppliers represent concentration risk.")
        
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
    "canonical_supplier_name",
    COUNT(DISTINCT "source_system") AS num_systems,
    LISTAGG(DISTINCT "source_system", ', ') WITHIN GROUP (ORDER BY "source_system") AS systems,
    COUNT(*) AS transaction_count,
    SUM(TO_NUMBER("line_amount_usd")) AS total_spend,
    ROUND(AVG(TO_NUMBER("match_confidence")), 1) AS avg_confidence
FROM {SPEND_TABLE}
WHERE "canonical_supplier_name" IS NOT NULL
GROUP BY "canonical_supplier_name"
HAVING COUNT(DISTINCT "source_system") > 1
ORDER BY total_spend DESC
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
        
        total_leakage = sum(float(r.get("TOTAL_SPEND", r.get("total_spend", 0)) or 0) for r in rows)
        max_spend = max(float(r.get("TOTAL_SPEND", r.get("total_spend", 0)) or 0) for r in rows) if rows else 0
        
        # Find top opportunities
        top3 = rows[:3]
        top3_spend = sum(float(r.get("TOTAL_SPEND", r.get("total_spend", 0)) or 0) for r in top3)
        
        # Find 3+ system suppliers (highest priority)
        multi_system = [r for r in rows if int(r.get("NUM_SYSTEMS", r.get("num_systems", 0)) or 0) >= 3]
        
        lines = []
        lines.append("💰 **Spend Consolidation Opportunities**\n")
        lines.append(f"**Total Addressable Spend:** {format_currency(total_leakage)} across {len(rows)} suppliers\n")
        lines.append("Suppliers being used across multiple ERP systems — opportunity for contract consolidation:\n")
        
        lines.append("| Supplier | Systems | Spend | |")
        lines.append("| --- | --- | --- | --- |")
        
        for row in rows:
            supplier = str(row.get("CANONICAL_SUPPLIER_NAME", row.get("canonical_supplier_name", "Unknown")))[:25]
            num_sys = row.get("NUM_SYSTEMS", row.get("num_systems", 0))
            spend = float(row.get("TOTAL_SPEND", row.get("total_spend", 0)) or 0)
            bar = make_bar(spend, max_spend, 8, "green")
            
            sys_indicator = "🔴" if int(num_sys or 0) >= 3 else "🟡"
            lines.append(f"| {supplier} | {sys_indicator} {num_sys} | {format_currency(spend)} | {bar} |")
        
        lines.append("")
        lines.append("**Legend:** 🔴 3+ systems (high priority) | 🟡 2 systems")
        
        # Smart recommendations
        lines.append("\n---\n**🎯 Recommended Actions:**\n")
        
        if rows:
            top = rows[0]
            top_name = top.get("CANONICAL_SUPPLIER_NAME", top.get("canonical_supplier_name", "Unknown"))
            top_spend = float(top.get("TOTAL_SPEND", top.get("total_spend", 0)) or 0)
            top_systems = top.get("SYSTEMS", top.get("systems", ""))
            est_savings = top_spend * 0.08
            lines.append(f"**1. Biggest Opportunity:** {top_name} ({format_currency(top_spend)} across {top_systems})")
            lines.append(f"   → Consolidate under single contract. Est. savings: {format_currency(est_savings)} (8% volume discount)")
        
        if multi_system:
            lines.append(f"\n**2. High Complexity:** {len(multi_system)} suppliers span 3+ systems:")
            for r in multi_system[:3]:
                lines.append(f"   • {r.get('CANONICAL_SUPPLIER_NAME', r.get('canonical_supplier_name', 'Unknown'))}: {r.get('SYSTEMS', r.get('systems', ''))}")
        
        if len(rows) >= 3:
            lines.append(f"\n**3. Quick Win:** Top 3 suppliers = {format_currency(top3_spend)} ({top3_spend/total_leakage*100:.0f}% of leakage)")
        
        total_est_savings = total_leakage * 0.05
        lines.append(f"\n**💵 Total Savings Potential:** {format_currency(total_est_savings)} (5% through consolidation)")
        lines.append(f"\n**Next:** Run `savings opportunities` to find price variances for these suppliers.")
        
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
    "category_name",
    COUNT(*) AS transaction_count,
    SUM(TO_NUMBER("line_amount_usd")) AS total_spend,
    COUNT(DISTINCT "canonical_supplier_name") AS unique_suppliers,
    ROUND(SUM(CASE WHEN "contract_reference" IS NULL THEN TO_NUMBER("line_amount_usd") ELSE 0 END) / 
          NULLIF(SUM(TO_NUMBER("line_amount_usd")), 0) * 100, 1) AS pct_off_contract
FROM {SPEND_TABLE}
GROUP BY "category_name"
ORDER BY total_spend DESC'''

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
        
        total_spend = sum(float(r.get("TOTAL_SPEND", r.get("total_spend", 0)) or 0) for r in rows)
        max_spend = max(float(r.get("TOTAL_SPEND", r.get("total_spend", 0)) or 0) for r in rows) if rows else 0
        high_risk_categories = [r for r in rows if float(r.get("PCT_OFF_CONTRACT", r.get("pct_off_contract", 0)) or 0) > 50]
        
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
            category = str(row.get("CATEGORY_NAME", row.get("category_name", "Unknown")))[:25]
            spend = float(row.get("TOTAL_SPEND", row.get("total_spend", 0)) or 0)
            bar = make_bar(spend, max_spend, 8, "purple")
            pct_off = float(row.get("PCT_OFF_CONTRACT", row.get("pct_off_contract", 0)) or 0)
            
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
    "canonical_supplier_name",
    "category_name",
    COUNT(DISTINCT "business_unit") AS num_bus,
    MIN(TO_NUMBER("line_amount_usd")) AS min_price,
    MAX(TO_NUMBER("line_amount_usd")) AS max_price,
    ROUND(AVG(TO_NUMBER("line_amount_usd")), 2) AS avg_price,
    ROUND((MAX(TO_NUMBER("line_amount_usd")) - MIN(TO_NUMBER("line_amount_usd"))) / NULLIF(AVG(TO_NUMBER("line_amount_usd")), 0) * 100, 1) AS variance_pct,
    SUM(TO_NUMBER("line_amount_usd")) AS total_spend,
    COUNT(*) AS transaction_count
FROM {SPEND_TABLE}
WHERE "canonical_supplier_name" IS NOT NULL
GROUP BY "canonical_supplier_name", "category_name"
HAVING COUNT(DISTINCT "business_unit") > 1 
   AND MAX(TO_NUMBER("line_amount_usd")) > MIN(TO_NUMBER("line_amount_usd")) * 1.2
ORDER BY (MAX(TO_NUMBER("line_amount_usd")) - MIN(TO_NUMBER("line_amount_usd"))) * COUNT(*) DESC
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
        
        total_addressable = sum(float(r.get("TOTAL_SPEND", r.get("total_spend", 0)) or 0) for r in rows)
        est_savings = total_addressable * 0.05
        
        lines = []
        lines.append("💵 **Savings Opportunities — Price Variance Analysis**\n")
        lines.append(f"**Addressable Spend:** {format_currency(total_addressable)}")
        lines.append(f"**Estimated Savings Potential:** {format_currency(est_savings)} (5% of addressable)\n")
        lines.append("Same supplier charging different prices across business units:\n")
        
        lines.append("| Supplier | Category | BUs | Min | Max | Variance |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        
        for row in rows[:12]:
            supplier = str(row.get("CANONICAL_SUPPLIER_NAME", row.get("canonical_supplier_name", "")))[:20]
            category = str(row.get("CATEGORY_NAME", row.get("category_name", "")))[:18]
            num_bus = row.get("NUM_BUS", row.get("num_bus", 0))
            min_price = float(row.get("MIN_PRICE", row.get("min_price", 0)) or 0)
            max_price = float(row.get("MAX_PRICE", row.get("max_price", 0)) or 0)
            variance = float(row.get("VARIANCE_PCT", row.get("variance_pct", 0)) or 0)
            
            if variance >= 50:
                var_display = f"🔴 {variance:.0f}%"
            elif variance >= 30:
                var_display = f"🟠 {variance:.0f}%"
            else:
                var_display = f"🟡 {variance:.0f}%"
            
            lines.append(f"| {supplier} | {category} | {num_bus} | {format_currency(min_price)} | {format_currency(max_price)} | {var_display} |")
        
        # Smart recommendations
        lines.append("\n---\n**🎯 Recommended Actions:**\n")
        
        # Get top opportunity details
        if rows:
            top_row = rows[0]
            top_supplier = top_row.get("CANONICAL_SUPPLIER_NAME", top_row.get("canonical_supplier_name", "Unknown"))
            top_category = top_row.get("CATEGORY_NAME", top_row.get("category_name", "Unknown"))
            top_variance = float(top_row.get("VARIANCE_PCT", top_row.get("variance_pct", 0)) or 0)
            top_min = float(top_row.get("MIN_PRICE", top_row.get("min_price", 0)) or 0)
            top_max = float(top_row.get("MAX_PRICE", top_row.get("max_price", 0)) or 0)
            
            lines.append(f"**1. Start Here:** {top_supplier} / {top_category}")
            lines.append(f"   → Price ranges from {format_currency(top_min)} to {format_currency(top_max)} ({top_variance:.0f}% variance)")
            lines.append(f"   → Action: Negotiate standardized pricing at {format_currency((top_min + top_max) / 2)} or lower")
        
        # Find highest variance
        if len(rows) > 1:
            highest_var_row = max(rows, key=lambda r: float(r.get("VARIANCE_PCT", r.get("variance_pct", 0)) or 0))
            highest_var = float(highest_var_row.get("VARIANCE_PCT", highest_var_row.get("variance_pct", 0)) or 0)
            highest_var_supplier = highest_var_row.get("CANONICAL_SUPPLIER_NAME", highest_var_row.get("canonical_supplier_name", "Unknown"))
            if highest_var_supplier != top_supplier and highest_var > 100:
                lines.append(f"\n**2. Biggest Variance:** {highest_var_supplier} has {highest_var:.0f}% price variance — suggests ad-hoc purchasing")
        
        lines.append(f"\n**3. Process Fix:** Implement standard pricing sheets for top suppliers. Flag POs exceeding catalog price by >10%.")
        lines.append(f"\n**💵 Bottom Line:** Standardizing prices could save {format_currency(est_savings)} annually.")
        lines.append(f"\n**Next:** Run `spend leakage` to see if these suppliers are also fragmented across systems.")
        
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
    "category_name",
    COUNT(DISTINCT "canonical_supplier_name") AS supplier_count,
    SUM(TO_NUMBER("line_amount_usd")) AS total_spend,
    MAX("canonical_supplier_name") AS primary_supplier
FROM {SPEND_TABLE}
WHERE "canonical_supplier_name" IS NOT NULL
GROUP BY "category_name"
HAVING COUNT(DISTINCT "canonical_supplier_name") = 1 AND SUM(TO_NUMBER("line_amount_usd")) > 100000
ORDER BY total_spend DESC
FETCH FIRST 10 ROWS ONLY'''

        sql_reliance = f'''SELECT 
    "canonical_supplier_name",
    SUM(TO_NUMBER("line_amount_usd")) AS supplier_spend,
    COUNT(*) AS transaction_count,
    COUNT(DISTINCT "category_name") AS categories_served,
    ROUND(SUM(TO_NUMBER("line_amount_usd")) / (SELECT SUM(TO_NUMBER("line_amount_usd")) FROM {SPEND_TABLE}) * 100, 1) AS pct_of_total
FROM {SPEND_TABLE}
WHERE "canonical_supplier_name" IS NOT NULL
GROUP BY "canonical_supplier_name"
HAVING SUM(TO_NUMBER("line_amount_usd")) / (SELECT SUM(TO_NUMBER("line_amount_usd")) FROM {SPEND_TABLE}) > 0.05
ORDER BY supplier_spend DESC
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
            single_source_spend = sum(float(r.get("TOTAL_SPEND", r.get("total_spend", 0)) or 0) for r in concentration_rows)
            lines.append(f"**🔴 Single-Source Categories:** {len(concentration_rows)} categories ({format_currency(single_source_spend)} at risk)\n")
            lines.append("| Category | Sole Supplier | Spend |")
            lines.append("| --- | --- | --- |")
            for row in concentration_rows[:6]:
                cat = str(row.get("CATEGORY_NAME", row.get("category_name", "")))[:22]
                supplier = str(row.get("PRIMARY_SUPPLIER", row.get("primary_supplier", "")))[:22]
                spend = float(row.get("TOTAL_SPEND", row.get("total_spend", 0)) or 0)
                lines.append(f"| {cat} | {supplier} | {format_currency(spend)} |")
            lines.append("")
        else:
            lines.append("✅ **Single-Source:** No high-value single-source categories detected.\n")
        
        if reliance_rows:
            lines.append("**🟠 Supplier Concentration:**\n")
            lines.append("| Supplier | Spend | % of Total | Categories |")
            lines.append("| --- | --- | --- | --- |")
            max_spend = max(float(r.get("SUPPLIER_SPEND", r.get("supplier_spend", 0)) or 0) for r in reliance_rows) if reliance_rows else 0
            for row in reliance_rows[:8]:
                supplier = str(row.get("CANONICAL_SUPPLIER_NAME", row.get("canonical_supplier_name", "")))[:22]
                spend = float(row.get("SUPPLIER_SPEND", row.get("supplier_spend", 0)) or 0)
                pct = float(row.get("PCT_OF_TOTAL", row.get("pct_of_total", 0)) or 0)
                cats = row.get("CATEGORIES_SERVED", row.get("categories_served", 0))
                bar = make_bar(spend, max_spend, 6, "orange")
                lines.append(f"| {supplier} | {format_currency(spend)} | {pct:.1f}% {bar} | {cats} |")
            lines.append("")
        else:
            lines.append("✅ **Concentration:** No suppliers exceed 5% of total spend.\n")
        
        # Smart recommendations
        lines.append("---\n**🎯 Recommended Actions:**\n")
        
        if concentration_rows:
            top_single = concentration_rows[0]
            top_cat = top_single.get("CATEGORY_NAME", top_single.get("category_name", "Unknown"))
            top_sup = top_single.get("PRIMARY_SUPPLIER", top_single.get("primary_supplier", "Unknown"))
            top_spend = float(top_single.get("TOTAL_SPEND", top_single.get("total_spend", 0)) or 0)
            lines.append(f"**1. Critical Risk:** {top_cat} depends entirely on {top_sup} ({format_currency(top_spend)})")
            lines.append(f"   → Action: Identify 1-2 alternate suppliers within 30 days")
            lines.append(f"   → Interim: Review contract for early termination risk")
        
        if reliance_rows:
            top_rel = reliance_rows[0]
            top_name = top_rel.get("CANONICAL_SUPPLIER_NAME", top_rel.get("canonical_supplier_name", "Unknown"))
            top_spend = float(top_rel.get("SUPPLIER_SPEND", top_rel.get("supplier_spend", 0)) or 0)
            top_pct = float(top_rel.get("PCT_OF_TOTAL", top_rel.get("pct_of_total", 0)) or 0)
            lines.append(f"\n**2. Leverage Opportunity:** {top_name} = {top_pct:.1f}% of spend ({format_currency(top_spend)})")
            lines.append(f"   → Action: Schedule QBR to negotiate better terms (volume discount, payment terms)")
            lines.append(f"   → Your volume justifies preferred pricing tier")
        
        if concentration_rows and len(concentration_rows) >= 3:
            lines.append(f"\n**3. Diversification:** {len(concentration_rows)} single-source categories need backup suppliers")
            lines.append(f"   → Target: No category >$500K should be single-source")
        
        lines.append(f"\n**Next:** Run `maverick spend` to check if high-risk suppliers also have contract gaps.")
        
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
    SUBSTR("po_date", 1, 7) AS month,
    COUNT(*) AS transaction_count,
    SUM(TO_NUMBER("line_amount_usd")) AS total_spend,
    COUNT(DISTINCT "canonical_supplier_name") AS active_suppliers,
    ROUND(SUM(CASE WHEN "contract_reference" IS NULL THEN TO_NUMBER("line_amount_usd") ELSE 0 END) / 
          NULLIF(SUM(TO_NUMBER("line_amount_usd")), 0) * 100, 1) AS pct_maverick
FROM {SPEND_TABLE}
WHERE "po_date" IS NOT NULL
GROUP BY SUBSTR("po_date", 1, 7)
ORDER BY month DESC
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
            return {"messages": [{"role": "ai", "content": "No trend data available. Check that po_date is populated."}]}
        
        rows = list(reversed(rows))
        
        total_spend = sum(float(r.get("TOTAL_SPEND", r.get("total_spend", 0)) or 0) for r in rows)
        avg_monthly = total_spend / len(rows) if rows else 0
        max_spend = max(float(r.get("TOTAL_SPEND", r.get("total_spend", 0)) or 0) for r in rows) if rows else 0
        
        if len(rows) >= 2:
            first_half = sum(float(r.get("TOTAL_SPEND", r.get("total_spend", 0)) or 0) for r in rows[:len(rows)//2])
            second_half = sum(float(r.get("TOTAL_SPEND", r.get("total_spend", 0)) or 0) for r in rows[len(rows)//2:])
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
        lines.append(f"**Period:** {rows[0].get('MONTH', rows[0].get('month', ''))} to {rows[-1].get('MONTH', rows[-1].get('month', ''))}")
        lines.append(f"**Total:** {format_currency(total_spend)} | **Avg Monthly:** {format_currency(avg_monthly)} | **Trend:** {trend}\n")
        
        lines.append("| Month | Spend | | Maverick % |")
        lines.append("| --- | --- | --- | --- |")
        
        for row in rows:
            month = row.get("MONTH", row.get("month", ""))
            spend = float(row.get("TOTAL_SPEND", row.get("total_spend", 0)) or 0)
            bar = make_bar(spend, max_spend, 10, "blue")
            maverick_pct = float(row.get("PCT_MAVERICK", row.get("pct_maverick", 0)) or 0)
            
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