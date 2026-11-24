"""Normalize and extract numeric answers for the GSM8K benchmark."""

from decimal import Decimal, InvalidOperation
import re


def numeric(value):
    try:
        return format(Decimal(value.replace(",", "").strip()).normalize(), "f")
    except (InvalidOperation, AttributeError):
        return None


def final_number(text):
    final = text.rsplit("</think>", 1)[-1]
    # Select the last balanced boxed expression, allowing nested LaTeX braces.
    candidates = []
    for match in re.finditer(r"\\boxed\s*\{", final):
        start = match.end()
        depth = 1
        for end in range(start, len(final)):
            depth += (final[end] == "{") - (final[end] == "}")
            if depth == 0:
                candidates.append(final[start:end])
                break
    content = candidates[-1] if candidates else final
    fraction = re.fullmatch(r"\s*\\(?:dfrac|frac)\{([-+]?\d[\d,]*(?:\.\d+)?)\}\{([-+]?\d[\d,]*(?:\.\d+)?)\}\s*", content)
    if fraction:
        denominator = Decimal(fraction.group(2).replace(",", ""))
        if denominator:
            return format((Decimal(fraction.group(1).replace(",", ""))/denominator).normalize(), "f")
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", content)
    return numeric(numbers[-1]) if numbers else None
