# ============================================================
# helpers.py — Formatting + Utility Functions
# ============================================================


def extract_text_from_last_message(state) -> str:
    """Extract text from last message — SpendAgent pattern."""
    last = state["messages"][-1]
    if hasattr(last, "content"):
        return str(last.content)
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(last)


def extract_rows_from_result(result) -> list:
    """
    Parse SQL tool result into list of dicts.
    Confirmed response shape: result.structuredContent.rows
    Normalizes column names to UPPERCASE.
    """
    if result is None:
        return []
    try:
        if isinstance(result, dict):
            payload = result.get("result", result)
            if isinstance(payload, dict):
                structured = payload.get("structuredContent", {})
                rows = structured.get("rows", [])
                if rows:
                    return [{k.upper(): v for k, v in row.items()} for row in rows]
                # Fallback: top-level rows
                rows = payload.get("rows", [])
                if rows:
                    return [{k.upper(): v for k, v in row.items()} for row in rows]
    except Exception:
        pass
    return []


def extract_rag_answer(result) -> str:
    """
    Parse RAG tool result into answer string.
    Confirmed response shape: result.structuredContent.answer
    """
    if result is None:
        return ""
    try:
        if isinstance(result, dict):
            payload = result.get("result", result)
            if isinstance(payload, dict):
                structured = payload.get("structuredContent", {})
                answer = structured.get("answer", "")
                if answer:
                    return answer
                # Fallback: text content
                content = payload.get("content", [])
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
    except Exception:
        pass
    return str(result)


def format_sqltool_result(result) -> str:
    """Format SQL result as markdown table. Used by utility nodes."""
    rows = extract_rows_from_result(result)
    if not rows:
        return "_No results returned._"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows[:50]:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def format_currency(val) -> str:
    if val is None:
        return "-"
    try:
        v = float(val)
        if v >= 1_000_000:
            return f"${v/1_000_000:,.1f}M"
        elif v >= 1_000:
            return f"${v:,.0f}"
        else:
            return f"${v:,.2f}"
    except:
        return str(val)


def make_bar(value, max_val, width=10, color="blue") -> str:
    colors = {"blue": "🟦", "green": "🟩", "orange": "🟧", "red": "🟥", "purple": "🟪"}
    block = colors.get(color, "🟦")
    try:
        filled = max(0, min(width, round((float(value) / float(max_val)) * width)))
        return block * filled + "⬜" * (width - filled)
    except:
        return "⬜" * width


def risk_color(score) -> str:
    try:
        s = int(score)
        if s >= 75: return "🔴"
        elif s >= 55: return "🟠"
        elif s >= 35: return "🟡"
        else: return "🟢"
    except:
        return "⚪"


def delivery_color(pct) -> str:
    try:
        p = float(pct)
        if p >= 95: return "🟢"
        elif p >= 85: return "🟡"
        elif p >= 70: return "🟠"
        else: return "🔴"
    except:
        return "⚪"
