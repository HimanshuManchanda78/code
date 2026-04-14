import ollama
import base64
import sys


def extract_text(image_path: str) -> str:
    """Extract text from an image using moondream vision model."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    response = ollama.chat(
        model="qwen2.5vl:7b",  # Using larger model for better analysis
        messages=[{
            "role": "user",
            "content": """You are a Senior Actuary at an insurance company like AON performing data validation.

CAREFULLY analyze this screenshot and look for these specific issues:

1. **Numerical Outliers**: Look at ALL numbers in tables. If one value is 10x higher or lower than similar values in the same column, FLAG IT.

2. **Chart Spikes**: In any line/bar chart, if one data point is dramatically higher or lower than others, FLAG IT.

3. **Data Inconsistencies**: Check if totals match the sum of parts. Check if averages make sense.

4. **Suspicious Patterns**: Any value that seems wrong compared to its neighbors.

READ EVERY NUMBER in the image. Compare each value to others in the same category.

OUTPUT FORMAT:
- First, list ALL numbers you see in the image
- Then, identify ANY value that looks unusual or suspicious
- For each issue found, explain WHY it's suspicious

If you find issues, start with: "ALERT - Issues detected:"
If everything looks normal, start with: "DATA VALIDATED - No issues found"

Be thorough and skeptical. Assume there ARE hidden errors to find.""",
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
