from pathlib import Path

import cv2

ROOT = Path("/root/autodl-tmp/CULane")
TRAIN_GT = ROOT / "list" / "train_gt.txt"
VAL_GT = ROOT / "list" / "val_gt.txt"
THRESHOLDS = [40, 50, 60, 70, 80, 90]


def img_from_line(line):
    return ROOT / line.split()[0].lstrip("/")


def mean_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read {path}")
    return float(img.mean())


def collect(path):
    values = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        values.append(mean_gray(img_from_line(line)))
    return values


def report(name, values):
    print(name, "total", len(values))
    for thr in THRESHOLDS:
        low = sum(v < thr for v in values)
        print(f"  < {thr}: {low}")


def main():
    train = collect(TRAIN_GT)
    val = collect(VAL_GT)
    report("train_gt", train)
    report("val_gt", val)


if __name__ == "__main__":
    main()
