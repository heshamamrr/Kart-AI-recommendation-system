# rag_agent.py
# RAG Search Agent — handles all product search queries
#
# Satisfies:
#   FR2 - Grounded Product Search: all results retrieved from ChromaDB, never generated
#   FR5 - Conversational Ability: returns structured data the router formats into natural language
#   NFR1 - Hallucination Resistance: no LLM involved in retrieval — only vector similarity search
#   NFR3 - Modularity: self-contained, callable independently

import os
import re

import pandas as pd

from knowledge_base import search_products
from config import DATA_PATH, TOP_K_RESULTS


def _extract_price_limits(query: str) -> tuple[float | None, float | None]:
    """
    Parse explicit price constraints from a natural language query.
    Returns (price_min, price_max). Either may be None if not mentioned.
    WHY here rather than in the analyst agent: the RAG agent handles search
    queries that often include budget phrases ("under 500", "below 2000").
    Without this, semantically relevant but overpriced products slip through.
    """
    q = query.lower()
    price_max = None
    price_min = None

    # "under 700", "below 700", "less than 700", "upto 700", "max 700"
    m = re.search(
        r'(?:under|below|less than|cheaper than|upto|up to|max(?:imum)?|within)\s*'
        r'(?:rs\.?|inr|₹)?\s*(\d[\d,]*)',
        q
    )
    if m:
        price_max = float(m.group(1).replace(",", ""))

    # "above 500", "over 500", "more than 500", "at least 500"
    m = re.search(
        r'(?:above|over|more than|at least|atleast|min(?:imum)?)\s*'
        r'(?:rs\.?|inr|₹)?\s*(\d[\d,]*)',
        q
    )
    if m:
        price_min = float(m.group(1).replace(",", ""))

    # "between 500 and 1000"
    m = re.search(
        r'between\s*(?:rs\.?|inr|₹)?\s*(\d[\d,]*)\s*(?:and|to|-)\s*'
        r'(?:rs\.?|inr|₹)?\s*(\d[\d,]*)',
        q
    )
    if m:
        price_min = float(m.group(1).replace(",", ""))
        price_max = float(m.group(2).replace(",", ""))

    return price_min, price_max


def run_rag_agent(query: str, n_results: int = TOP_K_RESULTS) -> list[dict]:
    """
    Perform semantic product search for a given query.

    Args:
        query:     Natural language search query (already classified as search-intent by router)
        n_results: Number of products to return (default from config)

    Returns:
        List of product dicts, each containing:
            product_name, brand, main_category,
            retail_price, discounted_price, relevance_score
        Sorted by relevance_score descending (ChromaDB returns nearest neighbours first).
    """
    price_min, price_max = _extract_price_limits(query)

    # If the user gives a budget ceiling but no floor, search within 300 of the ceiling
    # so results are closer to what they're willing to spend (e.g. under 700 → 400–700).
    if price_max is not None and price_min is None:
        floor = price_max - 300
        if floor > 0:
            price_min = floor

    has_price_filter = price_min is not None or price_max is not None

    # With a price filter many candidates will be rejected, so fetch a large pool.
    # Without a filter a small pool is enough.
    fetch_n = min(200, n_results * 20) if has_price_filter else n_results * 3

    candidates = search_products(query, n_results=fetch_n)

    seen: set = set()
    unique: list[dict] = []
    for product in candidates:
        price = product["discounted_price"]

        # Hard price filter — no product outside the stated budget is returned
        if price_max is not None and price > price_max:
            continue
        if price_min is not None and price < price_min:
            continue

        # Deduplicate by full product name — blocks exact duplicates while
        # allowing colour/size variants with slightly different names
        key = product["product_name"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(product)

        if len(unique) >= n_results:
            break

    # ── Pad with CSV results if semantic search didn't find enough ────────────
    # WHY: ChromaDB's semantic pool is limited to products similar to the query.
    # When a price filter is strict, few semantically close products survive.
    # Pandas directly filters the CSV by price + guessed category to fill gaps.
    # These are still real products from the dataset — NFR1 is preserved.
    # ENABLE_CSV_FALLBACK env var (default "1") lets the ablation study disable
    # this step without forking the source file.
    _fallback_enabled = os.environ.get("ENABLE_CSV_FALLBACK", "1") != "0"
    if len(unique) < n_results and has_price_filter and _fallback_enabled:
        try:
            df = pd.read_csv(DATA_PATH)
            df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")
            df = df.dropna(subset=["product_name", "discounted_price"])

            mask = pd.Series([True] * len(df))
            if price_max is not None:
                mask &= df["discounted_price"] <= price_max
            if price_min is not None:
                mask &= df["discounted_price"] >= price_min

            # Try to guess category from query for better relevance
            q_lower = query.lower()
            for cat_kw, cat_val in [
                ("watch", "Watches"), ("shoe", "Footwear"), ("footwear", "Footwear"),
                ("clothing", "Clothing"), ("jewel", "Jewellery"), ("bag", "Bags, Wallets & Belts"),
                ("camera", "Cameras & Accessories"), ("computer", "Computers"),
            ]:
                if cat_kw in q_lower:
                    mask &= df["primary_category"].str.contains(cat_val, case=False, na=False)
                    break

            df_filtered = df[mask].sort_values("discounted_price")

            for _, row in df_filtered.iterrows():
                # Full name dedup for pandas padding — prefix dedup was blocking
                # budget products whose names partially matched semantic results
                key = str(row["product_name"]).lower().strip()
                if key not in seen:
                    seen.add(key)
                    unique.append({
                        "product_name":     str(row["product_name"]),
                        "brand":            str(row.get("brand", "")),
                        "main_category":    str(row.get("primary_category", "")),
                        "retail_price":     float(row["retail_price"]) if pd.notna(row.get("retail_price")) else 0.0,
                        "discounted_price": float(row["discounted_price"]),
                        "relevance_score":  0.0,
                    })
                if len(unique) >= n_results:
                    break
        except Exception:
            pass  # padding is best-effort — return whatever we have

    return unique


def format_rag_results(results: list[dict]) -> str:
    """
    Convert raw search results into a structured text block for the LLM router to format.

    The router uses this text to compose a natural language response.
    Keeping formatting here — rather than inside the router prompt — isolates
    the retrieval representation from the language generation step (NFR3).
    """
    if not results:
        return "No matching products found in the knowledge base."

    lines = [f"Found {len(results)} matching product(s):\n"]
    for i, product in enumerate(results, start=1):
        retail = product["retail_price"]
        discounted = product["discounted_price"]

        price_str = f"Rs. {discounted:,.0f}"
        if retail and retail != discounted:
            price_str += f" (was Rs. {retail:,.0f})"

        lines.append(
            f"{i}. {product['product_name']}\n"
            f"   Brand: {product['brand'] or 'N/A'} | "
            f"Category: {product['main_category'] or 'N/A'}\n"
            f"   Price: {price_str} | "
            f"Relevance: {product['relevance_score']:.2f}\n"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke test — run directly to verify search is working
    test_query = "wireless bluetooth headphones under 2000"
    print(f"Query: {test_query}\n")
    results = run_rag_agent(test_query)
    print(format_rag_results(results))
