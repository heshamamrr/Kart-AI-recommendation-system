


import re


def _extract_price_limits(query: str) -> tuple[float | None, float | None]:
    q = query.lower()
    price_max = None
    price_min = None

    m = re.search(
        r'(?:under|below|less than|cheaper than|upto|up to|max)\s*'
        r'(?:rs\.?|inr|₹)?\s*(\d[\d,]*)', q
    )
    if m:
        price_max = float(m.group(1).replace(",", ""))

    m = re.search(
        r'(?:above|over|more than|at least|min)\s*'
        r'(?:rs\.?|inr|₹)?\s*(\d[\d,]*)', q
    )
    if m:
        price_min = float(m.group(1).replace(",", ""))

    return price_min, price_max