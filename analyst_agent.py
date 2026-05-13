# analyst_agent.py
# Data Analyst Agent — handles all analytical queries over the product dataset
#
# Satisfies:
#   FR3 - Analytical Query Handling: price filtering, category comparisons, ranked results
#   FR4 - Query Routing: operates as the analytical branch of the routing architecture
#   NFR1 - Hallucination Resistance: LLM is used ONLY for intent parsing (JSON spec extraction),
#           never for data retrieval — all numbers and product names come from the CSV via Pandas
#   NFR3 - Modularity: self-contained, callable independently of other agents

import json
import re

import ollama
import pandas as pd

from config import DATA_PATH, LLM_MODEL, TOP_K_RESULTS

# ── Non-brand values that appear in the brand column but are not real brands ──
# The Flipkart CSV has many rows where "brand" contains a color, fit type,
# fabric, or generic placeholder rather than an actual brand name.
_NON_BRANDS: frozenset[str] = frozenset({
    # Generic placeholders
    "generic", "unbranded", "no brand", "noname", "n/a", "na", "",
    # Colors
    "black", "white", "pink", "red", "purple", "blue", "beige", "grey", "gray",
    "maroon", "multicolor", "multi color", "brown", "yellow", "green", "orange",
    "navy", "cream", "gold", "silver", "teal", "turquoise", "olive", "coral",
    "lavender", "violet", "indigo", "magenta", "khaki", "mustard", "burgundy",
    # Fit / cut descriptors
    "regular", "slim", "regular fit", "slim fit", "straight", "skinny",
    "relaxed", "loose", "comfort", "classic fit",
    # Fabric / material
    "cotton", "polyester", "linen", "silk", "wool", "denim", "jersey",
    # Style / occasion
    "printed", "solid", "striped", "checked", "plain", "casual", "formal",
    "ethnic", "party", "sports", "floral",
    # Gender / age
    "women", "men", "kids", "boys", "girls", "unisex",
})

# ── Dataset loaded once at module level ────────────────────────────────────────
# Loading on first use (lazy) avoids startup cost when the agent is not needed.
_df: pd.DataFrame | None = None


def get_dataframe() -> pd.DataFrame:
    """Return the product DataFrame, loading and cleaning it on first call."""
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
        _df = _df.dropna(subset=["product_name"])
        _df["discounted_price"] = pd.to_numeric(_df["discounted_price"], errors="coerce")
        _df["retail_price"] = pd.to_numeric(_df["retail_price"], errors="coerce")
        _df["brand"] = _df["brand"].fillna("").astype(str)
        _df["primary_category"] = _df["primary_category"].fillna("").astype(str)
    return _df


# ── Intent parsing via LLM ─────────────────────────────────────────────────────
# WHY use the LLM here: natural language price/category extraction is more robust
# via a small LLM call than with regex. The LLM produces ONLY a JSON spec —
# it never reads or generates data. All data operations are deterministic Pandas.

_PARSE_SYSTEM = (
    "You are a JSON extraction assistant. "
    "You extract structured query parameters from natural language. "
    "Always return valid JSON only. No markdown. No explanation."
)

_PARSE_TEMPLATE = """Extract query parameters from this e-commerce analytical query.

Available operations:
- "top_cheapest"  : return the N cheapest products (sort by discounted_price ASC)
- "top_expensive" : return the N most expensive products (sort by discounted_price DESC)
- "filter"        : filter by category/brand/price, return results sorted cheapest first
- "average_price" : compute the average discounted price for a category or brand
- "count"         : count how many products match the criteria
- "brand_count"   : show how many products each brand has (within a category if given)

OPERATION SELECTION RULES (follow strictly):
- Any query containing "brand", "brands", "which brand", "best brand", "top brand", "popular brand" → use "brand_count"
- Any query asking for cheapest/lowest/affordable → use "top_cheapest"
- Any query asking for most expensive/premium/luxury → use "top_expensive"
- Any query asking for average/mean price → use "average_price"
- Any query asking how many/count → use "count"
- Everything else → use "filter"

Known categories (use exact spelling if mentioned, else null):
Clothing, Jewellery, Footwear, Mobiles & Accessories, Automotive,
Home Decor & Festive Needs, Beauty and Personal Care, Home Furnishing,
Kitchen & Dining, Computers, Watches, Baby Care, Tools & Hardware,
Toys & School Supplies, Pens & Stationery, Bags Wallets & Belts,
Furniture, Sports & Fitness, Cameras & Accessories, Gaming, Electronics

Return ONLY this JSON:
{{
  "operation": "<one of the operations above>",
  "category": "<category string or null>",
  "brand": "<brand string or null>",
  "price_max": <number or null>,
  "price_min": <number or null>,
  "top_n": <integer, default 5>
}}

Query: "{query}"
"""


def _parse_query(query: str) -> dict:
    """
    Call the LLM to extract a structured spec from the user's analytical query.
    Falls back to a safe default spec if parsing fails.
    """
    prompt = _PARSE_TEMPLATE.format(query=query) + "\n/no_think"

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _PARSE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            think=False,
            options={"temperature": 0},
        )
        text = response.message.content.strip()

        # Strip markdown code fences if the LLM adds them despite instructions
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())

    except (json.JSONDecodeError, Exception):
        pass

    # Safe fallback — treat as a generic filter query
    return {
        "operation": "filter",
        "category": None,
        "brand": None,
        "price_max": None,
        "price_min": None,
        "top_n": TOP_K_RESULTS,
    }


# ── Pandas operations ──────────────────────────────────────────────────────────

def _apply_filters(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Apply category, brand, and price filters to the DataFrame."""
    filtered = df.copy()

    if spec.get("category"):
        filtered = filtered[
            filtered["primary_category"].str.contains(spec["category"], case=False, na=False)
        ]

    if spec.get("brand"):
        filtered = filtered[
            filtered["brand"].str.contains(spec["brand"], case=False, na=False)
        ]

    if spec.get("price_max") is not None:
        filtered = filtered[filtered["discounted_price"] <= spec["price_max"]]

    if spec.get("price_min") is not None:
        filtered = filtered[filtered["discounted_price"] >= spec["price_min"]]

    return filtered


def _to_product_list(df: pd.DataFrame) -> list[dict]:
    """Convert a filtered DataFrame slice to a list of product dicts."""
    products = []
    for _, row in df.iterrows():
        products.append({
            "product_name": str(row["product_name"]),
            "brand":        str(row["brand"]),
            "main_category": str(row["primary_category"]),
            "retail_price":     float(row["retail_price"])    if pd.notna(row["retail_price"])    else 0.0,
            "discounted_price": float(row["discounted_price"]) if pd.notna(row["discounted_price"]) else 0.0,
        })
    return products


def _label(spec: dict) -> str:
    """Build a readable label from the spec for use in summary strings."""
    parts = []
    if spec.get("brand"):
        parts.append(spec["brand"])
    if spec.get("category"):
        parts.append(spec["category"])
    if spec.get("price_max") is not None:
        parts.append(f"under Rs. {spec['price_max']:,.0f}")
    if spec.get("price_min") is not None:
        parts.append(f"above Rs. {spec['price_min']:,.0f}")
    return " ".join(parts) if parts else "all products"


# ── Main entry point ───────────────────────────────────────────────────────────

def run_analyst_agent(query: str) -> dict:
    """
    Execute an analytical query over the product DataFrame.

    Args:
        query: Natural language analytical query (already classified by the router)

    Returns:
        Dict with:
          - "summary":  str — human-readable summary of the result
          - "products": list[dict] — product rows (may be empty for aggregate results)
    """
    df = get_dataframe()
    spec = _parse_query(query)

    operation = spec.get("operation", "filter")
    top_n = max(1, int(spec.get("top_n") or TOP_K_RESULTS))

    filtered = _apply_filters(df, spec)

    if filtered.empty:
        return {
            "summary": f"No products found matching your criteria ({_label(spec)}).",
            "products": [],
        }

    # ── top_cheapest ──────────────────────────────────────────────────────────
    if operation == "top_cheapest":
        result_df = (
            filtered.dropna(subset=["discounted_price"])
            .sort_values("discounted_price", ascending=True)
            .head(top_n)
        )
        return {
            "summary": f"Top {len(result_df)} cheapest {_label(spec)}:",
            "products": _to_product_list(result_df),
        }

    # ── top_expensive ─────────────────────────────────────────────────────────
    elif operation == "top_expensive":
        result_df = (
            filtered.dropna(subset=["discounted_price"])
            .sort_values("discounted_price", ascending=False)
            .head(top_n)
        )
        return {
            "summary": f"Top {len(result_df)} most expensive {_label(spec)}:",
            "products": _to_product_list(result_df),
        }

    # ── average_price ─────────────────────────────────────────────────────────
    elif operation == "average_price":
        prices = filtered["discounted_price"].dropna()
        if prices.empty:
            return {"summary": f"No price data found for {_label(spec)}.", "products": []}
        return {
            "summary": (
                f"Average discounted price of {_label(spec)}: "
                f"Rs. {prices.mean():,.0f}  |  "
                f"Min: Rs. {prices.min():,.0f}  |  "
                f"Max: Rs. {prices.max():,.0f}  |  "
                f"({len(prices):,} products)"
            ),
            "products": [],
        }

    # ── count ─────────────────────────────────────────────────────────────────
    elif operation == "count":
        return {
            "summary": f"Found {len(filtered):,} {_label(spec)} products in the database.",
            "products": [],
        }

    # ── brand_count ───────────────────────────────────────────────────────────
    elif operation == "brand_count":
        real_brands = filtered[~filtered["brand"].str.strip().str.lower().isin(_NON_BRANDS)]
        counts = real_brands["brand"].value_counts().head(top_n)
        if counts.empty:
            return {"summary": f"No brand data found for {_label(spec)}.", "products": []}
        lines = [f"  {i+1}. {brand}: {cnt:,} products" for i, (brand, cnt) in enumerate(counts.items())]
        label = f" in {spec['category']}" if spec.get("category") else ""
        return {
            "summary": f"Top {len(counts)} brands{label} by product count:\n" + "\n".join(lines),
            "products": [],
        }

    # ── filter (default) ──────────────────────────────────────────────────────
    else:
        result_df = (
            filtered.dropna(subset=["discounted_price"])
            .sort_values("discounted_price", ascending=True)
            .head(top_n)
        )
        return {
            "summary": (
                f"Found {len(filtered):,} matching products ({_label(spec)}). "
                f"Showing {len(result_df)} cheapest:"
            ),
            "products": _to_product_list(result_df),
        }


# ── Formatting for the router ──────────────────────────────────────────────────

def format_analyst_results(result: dict) -> str:
    """
    Convert analyst results into a structured text block for the LLM router to format.
    Mirrors the format used by format_rag_results() in rag_agent.py for consistency.
    """
    lines = [result["summary"]]

    for i, p in enumerate(result.get("products", []), start=1):
        retail     = p["retail_price"]
        discounted = p["discounted_price"]
        price_str  = f"Rs. {discounted:,.0f}"
        if retail and retail != discounted:
            price_str += f" (was Rs. {retail:,.0f})"

        lines.append(
            f"\n{i}. {p['product_name']}\n"
            f"   Brand: {p['brand'] or 'N/A'} | "
            f"Category: {p['main_category'] or 'N/A'}\n"
            f"   Price: {price_str}"
        )

    return "\n".join(lines)


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What are the 5 cheapest watches?",
        "Show me clothing items under 500",
        "What is the average price of jewellery?",
        "Which brands have the most clothing products?",
        "How many footwear products are in the database?",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 50)
        result = run_analyst_agent(q)
        print(format_analyst_results(result))
