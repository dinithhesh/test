import json
import sys

try:
    with open("metrics.json") as f:
        data = json.load(f)
    
    # Set a low threshold for this test
    if data['accuracy'] >= 0.5:
        print("✅ Passed Threshold")
        sys.exit(0)
    else:
        print("❌ Accuracy too low")
        sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
