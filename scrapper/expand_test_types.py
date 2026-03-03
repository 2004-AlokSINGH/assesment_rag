"""
expand_test_types.py
--------------------
Reads shl_product_catalog.json, expands all Test Type letter codes
into their full human-readable names, and writes a new enriched file:
shl_product_catalog_expanded.json

Run from the scrapper/ folder:
    python expand_test_types.py

Why this matters for RAG:
  Before: Test Type = "K, P"
  After:  Test Type = "Knowledge & Skills, Personality & Behavior"
          Test Type Codes = ["K", "P"]          ← kept for filtering
          Test Type Full  = ["Knowledge & Skills", "Personality & Behavior"]

The richer text means the embedding model can match natural language like
"soft skills assessment" or "behavioral test" to the correct assessments
instead of relying on opaque letter codes.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Mapping: letter code → full name
# ---------------------------------------------------------------------------
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# ---------------------------------------------------------------------------
# Paths (works regardless of where you run from)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # RdsSys/
INPUT_PATH  = ROOT / "shl_product_catalog.json"
OUTPUT_PATH = ROOT / "shl_product_catalog_expanded.json"


def expand_test_types(raw_type_str: str) -> dict:
    """
    Given a raw string like "K, P" or "A, E, B, C, D, P",
    returns:
      {
        "codes": ["K", "P"],
        "full_names": ["Knowledge & Skills", "Personality & Behavior"],
        "display": "Knowledge & Skills, Personality & Behavior"
      }
    """
    codes = [t.strip() for t in raw_type_str.split(",") if t.strip()]
    full_names = [TEST_TYPE_MAP.get(c, c) for c in codes]
    return {
        "codes": codes,
        "full_names": full_names,
        "display": ", ".join(full_names),
    }


def main():
    print(f"Reading: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    total = len(catalog)
    expanded = 0

    for product in catalog:
        raw = product.get("Test Type", "")
        result = expand_test_types(raw)

        # Overwrite "Test Type" with full readable string for embeddings
        product["Test Type"]        = result["display"]
        # Keep original codes for structured filtering
        product["Test Type Codes"]  = result["codes"]
        # Also store full names as list for downstream use
        product["Test Type Full"]   = result["full_names"]

        if result["codes"]:
            expanded += 1

    print(f"Expanded test types for {expanded}/{total} products")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"Saved enriched catalog → {OUTPUT_PATH}")

    # Show a few samples
    print("\nSample transformations:")
    for p in catalog[:5]:
        print(f"  {p['Product Name'][:40]:<40} | {p['Test Type']}")


if __name__ == "__main__":
    main()
