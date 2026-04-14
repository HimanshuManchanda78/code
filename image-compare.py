import ollama
import base64
import sys


def extract_text(image_path: str) -> str:
    """Extract text from an image using moondream vision model."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    response = ollama.chat(
        model="minicpm-v",  # Faster 3B model - good balance of speed and accuracy
        messages=[{
            "role": "user",
            "content": """ROLE: You are a Senior Actuary at AON Insurance performing critical data quality review. 
Errors in insurance data can cost millions and damage AON's reputation. Be extremely thorough.

STEP 1 - READ ALL DATA:
List every number you see in tables, charts, and labels. Include:
- All values in data tables (row by row)
- All data points in charts/graphs
- All percentages shown
- All totals and subtotals

STEP 2 - VALIDATE EACH CHECK:

CHECK A - Column Consistency:
For each column in tables, list all values. Flag if ANY value is more than 3x different from others in same column.
Example: If most values are $4,000-$5,000 but one is $45,000, that's a RED FLAG.

CHECK B - Chart Anomalies:
In line/bar charts, identify the typical range. Flag any spike or dip that's more than 2x the average.
Example: If monthly values are around 2.0-2.5 but one month shows 8.5, that's a RED FLAG.

CHECK C - Mathematical Validation:
- Do percentages in pie charts add up to 100%?
- Do regional totals add up to the grand total shown?
- Is the average shown consistent with the individual values?

CHECK D - Loss Ratio Review:
Loss ratios in insurance typically range 55-75%. Flag any outside 50-80% range.

CHECK E - Cross-Reference:
Compare related metrics. If claims are high but payouts are low (or vice versa), flag it.

STEP 3 - OUTPUT FORMAT:

If issues found:
```
🚨 ALERT - DATA VALIDATION FAILED

Issue 1: [Specific location] - [Exact problematic value] vs [Expected range]
         WHY: [Explanation]

Issue 2: [Specific location] - [Exact problematic value] vs [Expected range]  
         WHY: [Explanation]

RECOMMENDATION: [What should be verified]
```

If no issues:
```
✅ DATA VALIDATED - All checks passed
Summary: [Brief confirmation of what was checked]
```

NOW ANALYZE THE IMAGE CAREFULLY. Read every single number.""",
            "images": [image_data]
        }]
    )
    return response["message"]["content"]


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "Screenshot_Expected.png"
    
    print(f"Extracting text from: {image_path}")
    print("-" * 40)
    
    text = extract_text(image_path)
    print(text)
