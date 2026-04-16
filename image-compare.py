"""
VisualVal - Actuarial Screenshot Validator
AON Insurance Quality Assurance Tool

Uses vision LLM to validate insurance application screenshots
for data accuracy, visual completeness, and actuarial reasonableness.
"""

import ollama
import base64
import sys
import os
from pathlib import Path
from datetime import datetime


# Configuration
DEFAULT_MODEL = "qwen2.5vl:7b"  # Vision model for detailed analysis
FAST_MODEL = "moondream"        # Fast lightweight model

# Prompts directory
PROMPTS_DIR = Path(__file__).parent / "prompts"


# ============================================================
# SYSTEM PROMPT - Defines the LLM's actuarial expertise
# ============================================================
SYSTEM_PROMPT = """You are a Senior Actuary at AON Insurance with 15+ years of experience in data quality assurance.

## YOUR EXPERTISE
- Insurance loss ratios, claims analysis, premium calculations
- Actuarial tables, mortality rates, risk assessments
- Financial reporting standards (GAAP, IFRS 17, Solvency II)
- Data visualization best practices for insurance metrics

## YOUR ROLE
You are performing a CRITICAL quality review of insurance application screenshots before release. Errors in actuarial data can cost millions and damage AON's reputation. You must be extremely thorough.

## VALIDATION FRAMEWORK

### 1. VISUAL INTEGRITY
- All chart labels present (axes, legends, titles)
- Data points properly rendered (no missing bars, lines, or segments)
- Color coding consistent with legend
- Text is readable, not truncated or overlapping

### 2. NUMERICAL ACCURACY
- Percentages sum to 100% where applicable
- Totals match sum of components
- Averages are mathematically consistent with data
- Year-over-year comparisons are directionally correct

### 3. ACTUARIAL REASONABLENESS
- Loss ratios: Typically 55-75% (flag if outside 45-85%)
- Combined ratios: Typically 95-105% (flag if outside 85-115%)
- Claim frequencies: Check for outliers (>2 standard deviations)
- Premium growth: Flag if >20% YoY without explanation

### 4. DATA ANOMALIES
- Identify values that are >3x different from peers in same category
- Spot sudden spikes/drops (>200% change) in time series
- Flag negative values where only positives expected
- Check for suspicious round numbers suggesting placeholders

## OUTPUT RULES
- Be specific: cite exact cell location, chart position, or data point
- Quantify issues: state the actual value vs expected range
- Prioritize: Critical (data errors) > Major (missing elements) > Minor (cosmetic)
- Be actionable: explain what needs verification or correction"""


# ============================================================
# USER PROMPT - Defines the task to perform
# ============================================================
USER_PROMPT = """Analyze this insurance application screenshot for data quality issues.

## REQUIRED CHECKS

1. **Read ALL visible data**: Tables, charts, labels, legends, totals
2. **Validate mathematical accuracy**: Do totals add up? Are percentages correct?
3. **Check for anomalies**: Any values that seem out of range for insurance data?
4. **Verify visual completeness**: Any missing labels, cut-off text, or rendering issues?

## OUTPUT FORMAT

### If issues found:
```
🚨 VALIDATION FAILED

CRITICAL ISSUES:
• [Location] - [Problem]: Found [X], expected [Y]

MAJOR ISSUES:
• [Location] - [Problem]: [Description]

MINOR ISSUES:
• [Location] - [Problem]: [Description]

RECOMMENDED ACTIONS:
1. [Action item]
2. [Action item]
```

### If no issues:
```
✅ VALIDATION PASSED

Checked:
• [List of elements validated]
• [Data ranges verified]

No anomalies detected.
```

Now analyze the screenshot:"""


def encode_image(image_path: str) -> str:
    """Encode image to base64 for API transmission."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def validate_screenshot(
    image_path: str,
    model: str = DEFAULT_MODEL,
    context: str = None,
    verbose: bool = True
) -> dict:
    """
    Validate an insurance application screenshot.
    
    Args:
        image_path: Path to the screenshot file
        model: Ollama model to use for analysis
        context: Optional additional context about the screenshot
        verbose: Whether to print progress messages
    
    Returns:
        Dictionary containing validation results
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if verbose:
        print(f"🔍 Analyzing: {image_path}")
        print(f"📦 Model: {model}")
        print("-" * 50)
    
    # Build user prompt with optional context
    user_content = USER_PROMPT
    if context:
        user_content = f"CONTEXT: {context}\n\n{user_content}"
    
    # Encode image
    image_data = encode_image(image_path)
    
    # Call the vision model with proper message structure
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content,
                "images": [image_data]
            }
        ],
        options={
            "temperature": 0.1,  # Low temperature for consistent analysis
            "num_predict": 2048  # Allow detailed response
        }
    )
    
    result = {
        "image_path": image_path,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "analysis": response["message"]["content"],
        "passed": "✅" in response["message"]["content"] or "PASSED" in response["message"]["content"].upper()
    }
    
    if verbose:
        print(result["analysis"])
        print("-" * 50)
        status = "✅ PASSED" if result["passed"] else "❌ ISSUES FOUND"
        print(f"\nStatus: {status}")
    
    return result


# CLI Interface
if __name__ == "__main__":
    # Parse arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default to the sample image
        image_path = "ActualScreenshots/AppScreenshot.png"
    
    # Check for model flag
    model = DEFAULT_MODEL
    if "--fast" in sys.argv:
        model = FAST_MODEL
        print("⚡ Using fast model for quick analysis")
    
    try:
        validate_screenshot(image_path, model=model)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
