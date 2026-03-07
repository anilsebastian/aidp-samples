# ============================================================
# helpers.py — Formatting and utility functions
# ============================================================
from datetime import datetime
from typing import Any, List

from config import CATALOG_KEY, SCHEMA_KEY


def now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def format_currency(val) -> str:
    """Format number as currency with $ and commas."""
    if val is None:
        return "-"
    try:
        num = float(val)
        if num >= 1_000_000:
            return f"${num/1_000_000:,.1f}M"
        elif num >= 1_000:
            return f"${num:,.0f}"
        else:
            return f"${num:,.2f}"
    except (ValueError, TypeError):
        return str(val)


def format_number(val) -> str:
    """Format number with commas."""
    if val is None:
        return "-"
    try:
        num = float(val)
        if num >= 1_000_000:
            return f"{num/1_000_000:,.1f}M"
        elif num >= 1_000:
            return f"{num:,.0f}"
        else:
            return str(int(num)) if num == int(num) else f"{num:.1f}"
    except (ValueError, TypeError):
        return str(val)


def make_bar(value: float, max_value: float, width: int = 15, color: str = "blue") -> str:
    """Create a text-based bar chart using colored emoji squares."""
    if max_value == 0:
        return ""
    ratio = min(value / max_value, 1.0)
    filled = int(ratio * width)
    empty = width - filled
    
    colors = {
        "blue": "🟦",
        "green": "🟩",
        "orange": "🟧",
        "red": "🟥",
        "purple": "🟪",
        "yellow": "🟨",
    }
    filled_char = colors.get(color, "🟦")
    empty_char = "⬜"
    
    return filled_char * filled + empty_char * empty


def risk_emoji(risk_type: str) -> str:
    """Return emoji for risk type."""
    if risk_type is None:
        return "⚪"
    risk_lower = str(risk_type).lower()
    if "no contract" in risk_lower and "unmatched" in risk_lower:
        return "🔴"
    elif "no contract" in risk_lower:
        return "🟠"
    elif "unmatched" in risk_lower:
        return "🟠"
    elif "low confidence" in risk_lower:
        return "🟡"
    else:
        return "⚪"


def extract_rows_from_result(result: Any) -> List[dict]:
    """Extract rows from SQLTool result."""
    if result is None:
        return []
    if isinstance(result, dict):
        payload = result
        if isinstance(payload.get("result"), dict):
            payload = payload["result"]
        if isinstance(payload.get("structuredContent"), dict):
            payload = payload["structuredContent"]
        rows = payload.get("rows", [])
        if isinstance(rows, list):
            return rows
    return []


def format_sqltool_result(result: Any, as_table: bool = True) -> str:
    """Best-effort formatting for SQLTool outputs — returns markdown table when possible."""
    if result is None:
        return "(empty result)"

    if isinstance(result, str):
        return result[:4000]

    if isinstance(result, dict):
        payload = result
        if isinstance(payload.get("result"), dict):
            payload = payload["result"]
        if isinstance(payload.get("structuredContent"), dict):
            payload = payload["structuredContent"]

        rows = payload.get("rows")
        cols = payload.get("columns")

        if rows is None:
            return str(result)[:1500]

        if not rows:
            return "(No rows returned)"

        # rows as list[dict] — format as markdown table
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            if as_table and rows:
                columns = list(rows[0].keys())
                lines = []
                lines.append("| " + " | ".join(columns) + " |")
                lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
                for row in rows[:50]:
                    values = []
                    for col in columns:
                        val = row.get(col, "")
                        if isinstance(val, (int, float)):
                            if val >= 1000:
                                val = f"{val:,.0f}"
                            else:
                                val = str(val)
                        else:
                            val = str(val) if val is not None else ""
                        values.append(val)
                    lines.append("| " + " | ".join(values) + " |")
                
                if len(rows) > 50:
                    lines.append(f"\n*... showing 50 of {len(rows)} rows*")
                return "\n".join(lines)
            else:
                lines = [str(r) for r in rows[:50]]
                if len(rows) > 50:
                    lines.append(f"... showing 50 of {len(rows)} rows")
                return "\n".join(lines)

        # rows as list[list]
        col_names: List[str] = []
        if isinstance(cols, list):
            for c in cols:
                if isinstance(c, dict) and "name" in c:
                    col_names.append(str(c["name"]))
                else:
                    col_names.append(str(c))

        if as_table and col_names and rows:
            lines = []
            lines.append("| " + " | ".join(col_names) + " |")
            lines.append("| " + " | ".join(["---"] * len(col_names)) + " |")
            for r in rows[:50]:
                values = []
                for val in r:
                    if isinstance(val, (int, float)):
                        if val >= 1000:
                            val = f"{val:,.0f}"
                        else:
                            val = str(val)
                    else:
                        val = str(val) if val is not None else ""
                    values.append(val)
                lines.append("| " + " | ".join(values) + " |")
            if len(rows) > 50:
                lines.append(f"\n*... showing 50 of {len(rows)} rows*")
            return "\n".join(lines)
        else:
            lines = []
            if col_names:
                lines.append(" | ".join(col_names))
                lines.append("-" * 60)
            for r in rows[:50]:
                try:
                    lines.append(" | ".join([str(x) for x in r]))
                except Exception:
                    lines.append(str(r))
            if len(rows) > 50:
                lines.append(f"... showing 50 of {len(rows)} rows")
            return "\n".join(lines)

    if isinstance(result, list):
        return str(result[:50])[:4000]

    return str(result)[:4000]
