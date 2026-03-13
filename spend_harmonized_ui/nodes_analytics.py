# ============================================================
# nodes_analytics.py — Analytics nodes (JSON output for UI)
# ============================================================
import json
import traceback
from langgraph.graph import MessagesState

from config import SPEND_TABLE
from helpers import extract_rows_from_result
from sql_tools import make_sql_tool


def json_response(data: dict) -> dict:
    """Wrap response data in standard message format."""
    return {"messages": [{"role": "ai", "content": json.dumps(data)}]}


def error_response(node_name: str, error: Exception, trace: str) -> dict:
    """Standard error response."""
    return json_response({
        "answer_type": "error",
        "error": str(error),
        "node": node_name,
        "trace": trace
    })


async def top_suppliers_node(state: MessagesState):
    """Aggregate spend by canonical supplier."""
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
            return json_response({
                "answer_type": "text",
                "content": "No supplier data found. Try `sample spend` to verify data exists."
            })
        
        # Build chart data
        chart_data = []
        for row in rows:
            supplier = row.get("SUPPLIER", row.get("supplier", "Unknown"))
            spend = float(row.get("SPEND_USD", row.get("spend_usd", 0)) or 0)
            chart_data.append({"label": supplier, "value": spend})
        
        total_spend = sum(d["value"] for d in chart_data)
        top_supplier = chart_data[0]["label"] if chart_data else "Unknown"
        top_spend = chart_data[0]["value"] if chart_data else 0
        top_pct = (top_spend / total_spend * 100) if total_spend > 0 else 0
        
        return json_response({
            "answer_type": "chart",
            "chart_type": "bar",
            "title": "Top 10 Suppliers by Spend",
            "data": chart_data,
            "summary": f"{top_supplier} leads with ${top_spend:,.0f} ({top_pct:.1f}% of top 10). Total across top 10: ${total_spend:,.0f}",
            "currency": "USD",
            "total": total_spend
        })
    except Exception as e:
        tb = traceback.format_exc()
        print("top_suppliers_node error:", e)
        return error_response("top_suppliers", e, tb)


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
            return json_response({
                "answer_type": "text",
                "content": "No maverick spend detected. All transactions have contracts and matched suppliers.",
                "status": "ok"
            })
        
        # Build table data and stats
        table_data = []
        risk_breakdown = {"No Contract": 0, "Unmatched Supplier": 0, "Low Confidence Match": 0, "No Contract + Unmatched Supplier": 0}
        total_maverick = 0
        
        for row in rows:
            risk_type = row.get("RISK_TYPE", row.get("risk_type", "Other"))
            amount = float(row.get("LINE_AMOUNT_USD", row.get("line_amount_usd", 0)) or 0)
            total_maverick += amount
            
            # Count by risk type
            for key in risk_breakdown:
                if key in risk_type:
                    risk_breakdown[key] += amount
            
            table_data.append({
                "po_id": row.get("PO_ID", row.get("po_id", "")),
                "supplier": row.get("SUPPLIER_NAME", row.get("supplier_name", "Unknown")),
                "category": row.get("CATEGORY_NAME", row.get("category_name", "")),
                "business_unit": row.get("BUSINESS_UNIT", row.get("business_unit", "")),
                "amount": amount,
                "risk_type": risk_type
            })
        
        # Top offender
        top_supplier = rows[0].get("SUPPLIER_NAME", rows[0].get("supplier_name", "Unknown"))
        top_amount = float(rows[0].get("LINE_AMOUNT_USD", rows[0].get("line_amount_usd", 0)) or 0)
        
        return json_response({
            "answer_type": "table",
            "title": "Maverick Spend Analysis",
            "data": table_data,
            "columns": ["po_id", "supplier", "category", "business_unit", "amount", "risk_type"],
            "summary": f"Found ${total_maverick:,.0f} in maverick spend across {len(rows)} transactions. Largest: {top_supplier} at ${top_amount:,.0f}.",
            "stats": {
                "total_flagged": total_maverick,
                "transaction_count": len(rows),
                "risk_breakdown": {k: v for k, v in risk_breakdown.items() if v > 0}
            },
            "recommendations": [
                f"Review {top_supplier} (${top_amount:,.0f}) — largest flagged purchase",
                "Negotiate blanket contracts for repeat offenders",
                "Require contract reference for POs over $10K"
            ]
        })
    except Exception as e:
        tb = traceback.format_exc()
        print("maverick_spend_node error:", e)
        return error_response("maverick_spend", e, tb)


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
            return json_response({
                "answer_type": "text",
                "content": "No spend leakage detected. Each supplier is being managed through a single system.",
                "status": "ok"
            })
        
        # Build table data
        table_data = []
        total_leakage = 0
        multi_system = []
        
        for row in rows:
            spend = float(row.get("TOTAL_SPEND", row.get("total_spend", 0)) or 0)
            num_sys = int(row.get("NUM_SYSTEMS", row.get("num_systems", 0)) or 0)
            total_leakage += spend
            
            entry = {
                "supplier": row.get("CANONICAL_SUPPLIER_NAME", row.get("canonical_supplier_name", "Unknown")),
                "num_systems": num_sys,
                "systems": row.get("SYSTEMS", row.get("systems", "")),
                "spend": spend,
                "transaction_count": row.get("TRANSACTION_COUNT", row.get("transaction_count", 0))
            }
            table_data.append(entry)
            
            if num_sys >= 3:
                multi_system.append(entry)
        
        est_savings = total_leakage * 0.05
        top = table_data[0] if table_data else {}
        
        return json_response({
            "answer_type": "table",
            "title": "Spend Consolidation Opportunities",
            "data": table_data,
            "columns": ["supplier", "num_systems", "systems", "spend", "transaction_count"],
            "summary": f"${total_leakage:,.0f} addressable spend across {len(rows)} suppliers used in multiple systems. Est. savings: ${est_savings:,.0f} through consolidation.",
            "stats": {
                "total_addressable": total_leakage,
                "supplier_count": len(rows),
                "multi_system_count": len(multi_system),
                "estimated_savings": est_savings
            },
            "recommendations": [
                f"Consolidate {top.get('supplier', 'top supplier')} (${top.get('spend', 0):,.0f} across {top.get('systems', 'multiple systems')})",
                f"{len(multi_system)} suppliers span 3+ systems — highest priority",
                "Target 5% savings through volume consolidation"
            ]
        })
    except Exception as e:
        tb = traceback.format_exc()
        print("spend_leakage_node error:", e)
        return error_response("spend_leakage", e, tb)


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
            return json_response({
                "answer_type": "text",
                "content": "No category data found."
            })
        
        # Build chart and table data
        chart_data = []
        table_data = []
        total_spend = 0
        high_risk = []
        
        for row in rows:
            category = row.get("CATEGORY_NAME", row.get("category_name", "Unknown"))
            spend = float(row.get("TOTAL_SPEND", row.get("total_spend", 0)) or 0)
            pct_off = float(row.get("PCT_OFF_CONTRACT", row.get("pct_off_contract", 0)) or 0)
            total_spend += spend
            
            chart_data.append({"label": category, "value": spend})
            
            risk_level = "high" if pct_off >= 75 else "medium" if pct_off >= 50 else "low" if pct_off >= 25 else "ok"
            
            table_data.append({
                "category": category,
                "spend": spend,
                "transaction_count": row.get("TRANSACTION_COUNT", row.get("transaction_count", 0)),
                "unique_suppliers": row.get("UNIQUE_SUPPLIERS", row.get("unique_suppliers", 0)),
                "pct_off_contract": pct_off,
                "risk_level": risk_level
            })
            
            if pct_off > 50:
                high_risk.append(category)
        
        return json_response({
            "answer_type": "chart",
            "chart_type": "bar",
            "title": "Spend by Category",
            "data": chart_data,
            "table_data": table_data,
            "columns": ["category", "spend", "transaction_count", "unique_suppliers", "pct_off_contract", "risk_level"],
            "summary": f"${total_spend:,.0f} across {len(rows)} categories. {len(high_risk)} categories have >50% off-contract spend.",
            "stats": {
                "total_spend": total_spend,
                "category_count": len(rows),
                "high_risk_count": len(high_risk)
            },
            "high_risk_categories": high_risk
        })
    except Exception as e:
        tb = traceback.format_exc()
        print("spend_by_category_node error:", e)
        return error_response("spend_by_category", e, tb)


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
            return json_response({
                "answer_type": "text",
                "content": "No significant price variances detected. Pricing appears consistent across business units.",
                "status": "ok"
            })
        
        # Build table data
        table_data = []
        total_addressable = 0
        
        for row in rows:
            spend = float(row.get("TOTAL_SPEND", row.get("total_spend", 0)) or 0)
            total_addressable += spend
            
            variance = float(row.get("VARIANCE_PCT", row.get("variance_pct", 0)) or 0)
            risk_level = "high" if variance >= 50 else "medium" if variance >= 30 else "low"
            
            table_data.append({
                "supplier": row.get("CANONICAL_SUPPLIER_NAME", row.get("canonical_supplier_name", "Unknown")),
                "category": row.get("CATEGORY_NAME", row.get("category_name", "")),
                "num_business_units": row.get("NUM_BUS", row.get("num_bus", 0)),
                "min_price": float(row.get("MIN_PRICE", row.get("min_price", 0)) or 0),
                "max_price": float(row.get("MAX_PRICE", row.get("max_price", 0)) or 0),
                "avg_price": float(row.get("AVG_PRICE", row.get("avg_price", 0)) or 0),
                "variance_pct": variance,
                "total_spend": spend,
                "risk_level": risk_level
            })
        
        est_savings = total_addressable * 0.05
        top = table_data[0] if table_data else {}
        
        return json_response({
            "answer_type": "table",
            "title": "Price Variance Savings Opportunities",
            "data": table_data,
            "columns": ["supplier", "category", "num_business_units", "min_price", "max_price", "variance_pct", "total_spend", "risk_level"],
            "summary": f"${total_addressable:,.0f} addressable spend with price variances. Est. savings: ${est_savings:,.0f} (5%).",
            "stats": {
                "total_addressable": total_addressable,
                "estimated_savings": est_savings,
                "opportunity_count": len(rows)
            },
            "recommendations": [
                f"Start with {top.get('supplier', 'top supplier')} / {top.get('category', 'category')} — {top.get('variance_pct', 0):.0f}% variance",
                "Negotiate standardized pricing across business units",
                "Flag POs exceeding catalog price by >10%"
            ]
        })
    except Exception as e:
        tb = traceback.format_exc()
        print("savings_opportunities_node error:", e)
        return error_response("savings_opportunities", e, tb)


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
        
        # Build single-source data
        single_source_data = []
        single_source_spend = 0
        for row in concentration_rows:
            spend = float(row.get("TOTAL_SPEND", row.get("total_spend", 0)) or 0)
            single_source_spend += spend
            single_source_data.append({
                "category": row.get("CATEGORY_NAME", row.get("category_name", "")),
                "supplier": row.get("PRIMARY_SUPPLIER", row.get("primary_supplier", "")),
                "spend": spend
            })
        
        # Build concentration data
        concentration_data = []
        for row in reliance_rows:
            concentration_data.append({
                "supplier": row.get("CANONICAL_SUPPLIER_NAME", row.get("canonical_supplier_name", "")),
                "spend": float(row.get("SUPPLIER_SPEND", row.get("supplier_spend", 0)) or 0),
                "pct_of_total": float(row.get("PCT_OF_TOTAL", row.get("pct_of_total", 0)) or 0),
                "categories_served": row.get("CATEGORIES_SERVED", row.get("categories_served", 0)),
                "transaction_count": row.get("TRANSACTION_COUNT", row.get("transaction_count", 0))
            })
        
        recommendations = []
        if single_source_data:
            top_single = single_source_data[0]
            recommendations.append(f"Critical: {top_single['category']} depends entirely on {top_single['supplier']} (${top_single['spend']:,.0f})")
        if concentration_data:
            top_conc = concentration_data[0]
            recommendations.append(f"Leverage: {top_conc['supplier']} = {top_conc['pct_of_total']:.1f}% of spend — negotiate better terms")
        if len(single_source_data) >= 3:
            recommendations.append(f"Diversify: {len(single_source_data)} single-source categories need backup suppliers")
        
        return json_response({
            "answer_type": "risk_analysis",
            "title": "Supplier Risk Analysis",
            "single_source": {
                "data": single_source_data,
                "total_at_risk": single_source_spend,
                "count": len(single_source_data)
            },
            "concentration": {
                "data": concentration_data,
                "count": len(concentration_data)
            },
            "summary": f"{len(single_source_data)} single-source categories (${single_source_spend:,.0f} at risk). {len(concentration_data)} suppliers exceed 5% of total spend.",
            "recommendations": recommendations
        })
    except Exception as e:
        tb = traceback.format_exc()
        print("supplier_risk_node error:", e)
        return error_response("supplier_risk", e, tb)


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
            return json_response({
                "answer_type": "text",
                "content": "No trend data available. Check that po_date is populated."
            })
        
        # Reverse to chronological order
        rows = list(reversed(rows))
        
        # Build chart data
        chart_data = []
        total_spend = 0
        
        for row in rows:
            month = row.get("MONTH", row.get("month", ""))
            spend = float(row.get("TOTAL_SPEND", row.get("total_spend", 0)) or 0)
            total_spend += spend
            
            chart_data.append({
                "month": month,
                "spend": spend,
                "transaction_count": row.get("TRANSACTION_COUNT", row.get("transaction_count", 0)),
                "active_suppliers": row.get("ACTIVE_SUPPLIERS", row.get("active_suppliers", 0)),
                "pct_maverick": float(row.get("PCT_MAVERICK", row.get("pct_maverick", 0)) or 0)
            })
        
        avg_monthly = total_spend / len(rows) if rows else 0
        
        # Determine trend
        if len(rows) >= 2:
            first_half = sum(d["spend"] for d in chart_data[:len(chart_data)//2])
            second_half = sum(d["spend"] for d in chart_data[len(chart_data)//2:])
            if second_half > first_half * 1.1:
                trend = "increasing"
            elif second_half < first_half * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return json_response({
            "answer_type": "chart",
            "chart_type": "line",
            "title": "Spend Trend Analysis",
            "data": chart_data,
            "x_axis": "month",
            "y_axis": "spend",
            "summary": f"${total_spend:,.0f} total spend from {chart_data[0]['month']} to {chart_data[-1]['month']}. Avg monthly: ${avg_monthly:,.0f}. Trend: {trend}.",
            "stats": {
                "total_spend": total_spend,
                "avg_monthly": avg_monthly,
                "trend": trend,
                "period_start": chart_data[0]["month"] if chart_data else "",
                "period_end": chart_data[-1]["month"] if chart_data else ""
            }
        })
    except Exception as e:
        tb = traceback.format_exc()
        print("spend_trend_node error:", e)
        return error_response("spend_trend", e, tb)
