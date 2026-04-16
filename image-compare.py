"""
VisualVal - Actuarial Screenshot Validator
AON Insurance Quality Assurance Tool

Uses vision LLM to validate insurance application screenshots
for data accuracy, visual completeness, and actuarial reasonableness.
Generates annotated comparison images highlighting issues.
"""

import ollama
import base64
import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


# Configuration
DEFAULT_MODEL = "qwen2.5vl:7b"  # Vision model for detailed analysis
FAST_MODEL = "moondream"        # Fast lightweight model
OUTPUT_DIR = Path(__file__).parent / "ValidationReports"

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

## CRITICAL OUTPUT REQUIREMENT
You MUST return your analysis in valid JSON format with bounding box coordinates for each issue.
The coordinates should be approximate percentages (0-100) of where the issue appears in the image.
- x: horizontal position from left edge (0=left, 100=right)
- y: vertical position from top edge (0=top, 100=bottom)
- width: approximate width of the issue area as percentage
- height: approximate height of the issue area as percentage"""


# ============================================================
# USER PROMPT - Defines the task to perform
# ============================================================
USER_PROMPT = """Analyze this insurance application screenshot for data quality issues.

## REQUIRED CHECKS

1. **Read ALL visible data**: Tables, charts, labels, legends, totals
2. **Validate mathematical accuracy**: Do totals add up? Are percentages correct?
3. **Check for anomalies**: Any values that seem out of range for insurance data?
4. **Verify visual completeness**: Any missing labels, cut-off text, or rendering issues?

## MANDATORY JSON OUTPUT FORMAT

You MUST respond with ONLY valid JSON in this exact structure (no other text before or after):

```json
{
  "validation_passed": false,
  "summary": "Brief one-line summary of findings",
  "issues": [
    {
      "id": 1,
      "severity": "CRITICAL",
      "location": "Top-right chart",
      "problem": "Missing Y-axis label",
      "details": "The bar chart showing premium distribution has no Y-axis label making it impossible to interpret values",
      "bbox": {"x": 60, "y": 10, "width": 35, "height": 40}
    },
    {
      "id": 2,
      "severity": "MAJOR",
      "location": "Data table row 3",
      "problem": "Anomalous loss ratio",
      "details": "Loss ratio of 145% is outside acceptable range (45-85%)",
      "bbox": {"x": 20, "y": 50, "width": 60, "height": 10}
    }
  ],
  "recommendations": [
    "Verify data source for loss ratio calculation",
    "Add missing axis labels to all charts"
  ]
}
```

## BBOX COORDINATE RULES
- x, y: Position as percentage from top-left (0-100)
- width, height: Size as percentage of image dimensions
- Be as accurate as possible in locating the issue

## SEVERITY LEVELS
- CRITICAL: Data errors that could cause financial/compliance issues
- MAJOR: Missing elements or significant visual problems
- MINOR: Cosmetic issues or minor inconsistencies

If no issues found, return:
```json
{
  "validation_passed": true,
  "summary": "All checks passed - no issues detected",
  "issues": [],
  "recommendations": []
}
```

NOW ANALYZE THE IMAGE AND RETURN ONLY JSON:"""


def encode_image(image_path: str) -> str:
    """Encode image to base64 for API transmission."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_llm_response(response_text: str) -> dict:
    """
    Parse JSON from LLM response, handling potential formatting issues.
    """
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if "```json" in text:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
    elif "```" in text:
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    
    # Return default structure if parsing fails
    return {
        "validation_passed": False,
        "summary": "Unable to parse LLM response - see raw output",
        "issues": [{
            "id": 1,
            "severity": "MAJOR",
            "location": "Full image",
            "problem": "LLM response parsing failed",
            "details": response_text[:300],
            "bbox": {"x": 5, "y": 5, "width": 90, "height": 90}
        }],
        "recommendations": ["Re-run validation"],
        "raw_response": response_text
    }


def get_font(size: int = 14):
    """Get a font for drawing text, with fallback."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf"
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    
    return ImageFont.load_default()


def draw_issues_on_image(image: Image.Image, issues: list) -> Image.Image:
    """Draw red bounding boxes and labels on the image for each issue."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    img_width, img_height = annotated.size
    
    severity_colors = {
        "CRITICAL": (255, 0, 0),      # Red
        "MAJOR": (255, 128, 0),       # Orange
        "MINOR": (255, 255, 0)        # Yellow
    }
    
    label_font = get_font(16)
    number_font = get_font(24)
    
    for issue in issues:
        bbox = issue.get("bbox", {})
        severity = issue.get("severity", "MAJOR")
        issue_id = issue.get("id", 0)
        
        # Convert percentage coordinates to pixels
        x = int(bbox.get("x", 0) / 100 * img_width)
        y = int(bbox.get("y", 0) / 100 * img_height)
        width = int(bbox.get("width", 10) / 100 * img_width)
        height = int(bbox.get("height", 10) / 100 * img_height)
        
        width = max(width, 50)
        height = max(height, 30)
        
        color = severity_colors.get(severity, (255, 0, 0))
        
        # Draw rectangle with thick border
        for i in range(3):
            draw.rectangle(
                [x - i, y - i, x + width + i, y + height + i],
                outline=color
            )
        
        # Draw issue number in a circle
        circle_radius = 15
        circle_x = x - circle_radius
        circle_y = y - circle_radius
        draw.ellipse(
            [circle_x - circle_radius, circle_y - circle_radius,
             circle_x + circle_radius, circle_y + circle_radius],
            fill=color,
            outline=(255, 255, 255)
        )
        
        # Draw number in circle
        draw.text(
            (circle_x - 6, circle_y - 12),
            str(issue_id),
            fill=(255, 255, 255),
            font=number_font
        )
        
        # Draw severity label below the box
        label = f"#{issue_id}: {severity}"
        draw.rectangle(
            [x, y + height + 2, x + 120, y + height + 22],
            fill=color
        )
        draw.text(
            (x + 5, y + height + 4),
            label,
            fill=(255, 255, 255),
            font=label_font
        )
    
    return annotated


def create_comparison_image(
    original_path: str,
    analysis_result: dict,
    output_path: str = None
) -> str:
    """
    Create a side-by-side comparison image with:
    - Left: Original screenshot
    - Right: Annotated screenshot with issues highlighted
    - Bottom: Issue legend/summary
    """
    original = Image.open(original_path).convert("RGB")
    orig_width, orig_height = original.size
    
    issues = analysis_result.get("issues", [])
    annotated = draw_issues_on_image(original, issues)
    
    # Calculate dimensions for the composite image
    padding = 20
    header_height = 60
    legend_height = max(150, len(issues) * 50 + 80)
    
    total_width = orig_width * 2 + padding * 3
    total_height = orig_height + header_height + legend_height + padding * 3
    
    # Create composite image with white background
    composite = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(composite)
    
    title_font = get_font(24)
    header_font = get_font(18)
    text_font = get_font(14)
    
    # Draw header
    status_color = (0, 150, 0) if analysis_result.get("validation_passed") else (200, 0, 0)
    status_text = "✓ VALIDATION PASSED" if analysis_result.get("validation_passed") else "✗ VALIDATION FAILED"
    
    draw.rectangle([0, 0, total_width, header_height], fill=status_color)
    draw.text(
        (padding, 15),
        f"VisualVal Report - {status_text}",
        fill=(255, 255, 255),
        font=title_font
    )
    draw.text(
        (total_width - 300, 20),
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fill=(255, 255, 255),
        font=text_font
    )
    
    # Draw section labels
    y_offset = header_height + padding
    
    draw.text(
        (padding + orig_width // 2 - 80, y_offset),
        "ORIGINAL",
        fill=(100, 100, 100),
        font=header_font
    )
    draw.text(
        (padding * 2 + orig_width + orig_width // 2 - 100, y_offset),
        "ISSUES HIGHLIGHTED",
        fill=(200, 0, 0),
        font=header_font
    )
    
    # Paste original and annotated images
    image_y = y_offset + 30
    composite.paste(original, (padding, image_y))
    composite.paste(annotated, (padding * 2 + orig_width, image_y))
    
    # Draw borders around images
    draw.rectangle(
        [padding - 2, image_y - 2, padding + orig_width + 2, image_y + orig_height + 2],
        outline=(200, 200, 200),
        width=2
    )
    draw.rectangle(
        [padding * 2 + orig_width - 2, image_y - 2,
         padding * 2 + orig_width * 2 + 2, image_y + orig_height + 2],
        outline=(200, 0, 0),
        width=2
    )
    
    # Draw legend section
    legend_y = image_y + orig_height + padding
    draw.rectangle(
        [padding, legend_y, total_width - padding, total_height - padding],
        outline=(200, 200, 200),
        width=1
    )
    
    draw.text(
        (padding + 10, legend_y + 10),
        "ISSUES SUMMARY",
        fill=(50, 50, 50),
        font=header_font
    )
    
    summary = analysis_result.get("summary", "No summary available")
    draw.text(
        (padding + 10, legend_y + 40),
        f"Summary: {summary}",
        fill=(80, 80, 80),
        font=text_font
    )
    
    severity_colors = {
        "CRITICAL": (255, 0, 0),
        "MAJOR": (255, 128, 0),
        "MINOR": (255, 200, 0)
    }
    
    issue_y = legend_y + 70
    for issue in issues[:5]:
        severity = issue.get("severity", "MAJOR")
        color = severity_colors.get(severity, (255, 0, 0))
        
        draw.rectangle(
            [padding + 10, issue_y, padding + 25, issue_y + 15],
            fill=color
        )
        
        issue_text = f"#{issue.get('id', '?')} [{severity}] {issue.get('location', 'Unknown')}: {issue.get('problem', 'Unknown issue')}"
        draw.text(
            (padding + 35, issue_y),
            issue_text[:100],
            fill=(50, 50, 50),
            font=text_font
        )
        
        details = issue.get("details", "")[:120]
        draw.text(
            (padding + 35, issue_y + 18),
            f"   → {details}",
            fill=(100, 100, 100),
            font=text_font
        )
        
        issue_y += 45
    
    if len(issues) > 5:
        draw.text(
            (padding + 35, issue_y),
            f"... and {len(issues) - 5} more issues",
            fill=(100, 100, 100),
            font=text_font
        )
    
    # Generate output path if not provided
    if output_path is None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = Path(original_path).stem
        output_path = str(OUTPUT_DIR / f"{original_name}_report_{timestamp}.png")
    
    composite.save(output_path, "PNG", quality=95)
    
    return output_path


def validate_screenshot(
    image_path: str,
    model: str = DEFAULT_MODEL,
    context: str = None,
    verbose: bool = True,
    output_path: str = None
) -> dict:
    """
    Validate an insurance application screenshot and generate a comparison report.
    
    Args:
        image_path: Path to the screenshot file
        model: Ollama model to use for analysis
        context: Optional additional context about the screenshot
        verbose: Whether to print progress messages
        output_path: Optional custom path for the output report image
    
    Returns:
        Dictionary containing validation results and path to generated report
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
    if verbose:
        print("🤖 Sending to LLM for analysis...")
    
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
    
    raw_response = response["message"]["content"]
    
    if verbose:
        print("📝 Raw LLM Response:")
        print(raw_response)
        print("-" * 50)
    
    # Parse the JSON response
    analysis = parse_llm_response(raw_response)
    
    if verbose:
        print(f"📊 Parsed {len(analysis.get('issues', []))} issues")
    
    # Generate the comparison report image
    if verbose:
        print("🖼️  Generating comparison report image...")
    
    report_path = create_comparison_image(image_path, analysis, output_path)
    
    # Prepare result
    result = {
        "image_path": image_path,
        "report_path": report_path,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "validation_passed": analysis.get("validation_passed", False),
        "summary": analysis.get("summary", ""),
        "issues": analysis.get("issues", []),
        "recommendations": analysis.get("recommendations", []),
        "raw_response": raw_response
    }
    
    if verbose:
        print("-" * 50)
        if result["validation_passed"]:
            print("✅ VALIDATION PASSED")
        else:
            print("❌ VALIDATION FAILED")
            print(f"\n📋 Summary: {result['summary']}")
            print(f"\n🔴 Issues Found: {len(result['issues'])}")
            for issue in result["issues"]:
                severity = issue.get("severity", "UNKNOWN")
                icon = "🔴" if severity == "CRITICAL" else "🟠" if severity == "MAJOR" else "🟡"
                print(f"   {icon} #{issue.get('id', '?')} [{severity}] {issue.get('location', '')}: {issue.get('problem', '')}")
        
        print(f"\n📄 Report saved to: {report_path}")
    
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
