from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "output"
ROOT_DST = PROJECT_ROOT / "data" / "HGLane"
CATEGORIES = {"snow", "rain", "fog", "night", "dusk"}

moved = 0
for old_path in SRC_DIR.glob("*.png"):
    new_name = old_path.name.replace("_00001_.png", "")
    category = new_name.split("_")[0]
    if category not in CATEGORIES:
        continue
    new_path = ROOT_DST / category / new_name
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_path), str(new_path))
    moved += 1
    print(new_path)

print(f"Done. moved={moved}")
