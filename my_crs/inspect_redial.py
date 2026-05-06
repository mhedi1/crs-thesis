# inspect_redial.py
import json

path = r"C:\Users\mhfou\Desktop\thesis_crs\baseline_repo\KBRD_project\KBRD\data\redial\test_data.jsonl"

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 2:
            break
        data = json.loads(line)
        print(f"\n{'='*50}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*50}")
        print(json.dumps(data, indent=2))