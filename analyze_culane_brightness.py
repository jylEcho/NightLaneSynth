from pathlib import Path
import random
import statistics

import cv2

ROOT = Path("/root/autodl-tmp/CULane")
SPLITS = {
    "test0_normal": ROOT / "list" / "test_split" / "test0_normal.txt",
    "test8_night": ROOT / "list" / "test_split" / "test8_night.txt",
    "train_gt": ROOT / "list" / "train_gt.txt",
    "val_gt": ROOT / "list" / "val_gt.txt",
}
SEED = 20260617
MAX_SAMPLES = {
    "test0_normal": 1200,
    "test8_night": 1200,
    "train_gt": 2000,
    "val_gt": 800,
}


def image_rel_from_line(line: str) -> str:
    parts = line.split()
    rel = parts[0] if parts else line.strip()
    return rel.lstrip("/")


def mean_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read {path}")
    return float(img.mean())


def summarize(name, values):
    values = sorted(values)
    def q(p: float) -> float:
        idx = min(len(values) - 1, max(0, int(p * (len(values) - 1))))
        return values[idx]
    return (
        f"{name} count={len(values)} min={values[0]:.2f} q10={q(0.10):.2f} "
        f"q25={q(0.25):.2f} median={q(0.50):.2f} q75={q(0.75):.2f} "
        f"q90={q(0.90):.2f} max={values[-1]:.2f} mean={statistics.mean(values):.2f}"
    )


def main():
    rng = random.Random(SEED)
    for name, list_path in SPLITS.items():
        lines = [x.strip() for x in list_path.read_text().splitlines() if x.strip()]
        if len(lines) > MAX_SAMPLES[name]:
            lines = rng.sample(lines, MAX_SAMPLES[name])
        values = [mean_gray(ROOT / image_rel_from_line(line)) for line in lines]
        print(summarize(name, values))


if __name__ == "__main__":
    main()
