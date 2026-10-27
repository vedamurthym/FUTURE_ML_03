import shutil
from pathlib import Path

src_dir = Path(r"C:\Users\SKC\Downloads\files")
dst_dir = Path(r"c:\Users\SKC\OneDrive\Documents\Desktop\TASK-3\data\test_resumes")
dst_dir.mkdir(parents=True, exist_ok=True)

for i in range(1, 21):
    fname = f"resume_{i:02d}.pdf"
    src = src_dir / fname
    dst = dst_dir / fname
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied {fname} ({src.stat().st_size} bytes)")
    else:
        print(f"Warning: {src} not found!")
