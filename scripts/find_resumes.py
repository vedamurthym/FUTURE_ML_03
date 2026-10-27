import os
from pathlib import Path

search_dirs = [
    Path(r"C:\Users\SKC\OneDrive\Documents\Desktop"),
    Path(r"C:\Users\SKC\Downloads"),
    Path(r"C:\Users\SKC\Documents"),
    Path(r"C:\Users\SKC\OneDrive\Documents"),
    Path(r"C:\Users\SKC\Desktop"),
]

found = []
for sdir in search_dirs:
    if not sdir.exists():
        continue
    try:
        for root, dirs, files in os.walk(sdir):
            for f in files:
                if f.lower().startswith("resume_") and f.lower().endswith(".pdf"):
                    found.append(os.path.join(root, f))
    except Exception as e:
        print(f"Error scanning {sdir}: {e}")

print(f"Found {len(found)} resumes:")
for f in found[:30]:
    print(f)
