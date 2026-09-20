import json
import os
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import requests

COMFYUI_API_URL = "http://localhost:8188"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "HGLane"
IMAGE_ROOT = DATA_ROOT / "normal"
CANNY_ROOT = DATA_ROOT / "canny"
ANNOTATOR_ROOT = DATA_ROOT / "normal"
OUTPUT_ROOT = PROJECT_ROOT / "output"
INPUT_ROOT = PROJECT_ROOT / "input"
JSON_FILE_PATH_CANNY_P2P = PROJECT_ROOT / "v11_canny_p2p.json"
JSON_FILE_PATH_CANNY = PROJECT_ROOT / "v11_canny.json"
LOW_THRESHOLD = 100
HIGH_THRESHOLD = 200
LABEL_SETTINGS = {
    "snow": {
        "seed": "5",
        "positive_prompt": "falling snow, daytime",
        "negative_prompt": "unrealistic proportions",
        "workflow": JSON_FILE_PATH_CANNY,
    },
    "rain": {
        "seed": "9",
        "positive_prompt": "falling rain, daytime",
        "negative_prompt": "Low-quality, distorted, unrealistic proportions, dull colors, out of focus, messy background, duplicate characters, foggy.",
        "workflow": JSON_FILE_PATH_CANNY,
    },
    "fog": {
        "seed": "30",
        "positive_prompt": "mist, daytime",
        "negative_prompt": "Low-quality, blurry, distorted, unrealistic proportions, dull colors, out of focus, messy background, duplicate characters.",
        "workflow": JSON_FILE_PATH_CANNY,
    },
    "night": {
        "seed": "190435371239247",
        "positive_prompt": "change the sky to night, but not change the lane. detailed, 4k",
        "negative_prompt": "Low-quality, blurry, distorted, unrealistic proportions, dull colors, out of focus, messy background, duplicate characters.",
        "workflow": JSON_FILE_PATH_CANNY_P2P,
    },
    "dusk": {
        "seed": "190435371239249",
        "positive_prompt": "change the sky to dusk, random add less sunset, but not change the lane. detailed, 4k",
        "negative_prompt": "Low-quality, blurry, distorted, unrealistic proportions, dull colors, out of focus, messy background, duplicate characters.",
        "workflow": JSON_FILE_PATH_CANNY_P2P,
    },
}


def load_json_file(file_path: Path):
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def upload_image(image_path: Path):
    with image_path.open("rb") as image_file:
        files = {"image": image_file}
        response = requests.post(
            f"{COMFYUI_API_URL}/upload/image", files=files, timeout=120
        )
    response.raise_for_status()
    return response.json()["name"]


def submit_prompt(json_data):
    data = {"client_id": "1", "prompt": json_data}
    response = requests.post(f"{COMFYUI_API_URL}/prompt", json=data, timeout=120)
    response.raise_for_status()
    return response.json()


class CannyDetector:
    def __call__(self, img, annotator_file, low_threshold, high_threshold):
        canny_edges = cv2.Canny(img, low_threshold, high_threshold)
        polygons = self.read_polygons_from_file(annotator_file)
        mask = np.zeros_like(canny_edges)
        for polygon in polygons:
            points = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [points], color=255)
        color_mask = self.get_color_mask(img)
        color_masked_edges = cv2.bitwise_and(mask, color_mask)
        return cv2.bitwise_or(canny_edges, color_masked_edges)

    @staticmethod
    def read_polygons_from_file(file_path):
        polygons = []
        with open(file_path, "r") as file:
            for line in file:
                points = line.strip().split()
                polygon = [
                    (float(points[i]), float(points[i + 1]))
                    for i in range(0, len(points), 2)
                ]
                polygons.append(polygon)
        return polygons

    @staticmethod
    def get_color_mask(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        return cv2.bitwise_or(white_mask, yellow_mask)


def ensure_dirs():
    for path in [CANNY_ROOT, OUTPUT_ROOT, INPUT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    for label in LABEL_SETTINGS:
        (DATA_ROOT / label).mkdir(parents=True, exist_ok=True)


def build_work_items():
    image_list = []
    canny_list = []
    annotator_file_list = []
    for root, _, files in os.walk(IMAGE_ROOT):
        for file in files:
            if not file.endswith(".jpg"):
                continue
            image_path = Path(root) / file
            annotator_path = ANNOTATOR_ROOT / file.replace(".jpg", ".lines.txt")
            canny_path = CANNY_ROOT / file.replace(".jpg", "_canny.png")
            if annotator_path.exists():
                image_list.append(image_path)
                canny_list.append(canny_path)
                annotator_file_list.append(annotator_path)
    image_list = sorted(
        image_list,
        key=lambda x: int(re.search(r"normal_(\d+)\.jpg", str(x)).group(1)),
    )
    canny_list = sorted(
        canny_list,
        key=lambda x: int(re.search(r"normal_(\d+)_canny\.png", str(x)).group(1)),
    )
    annotator_file_list = sorted(
        annotator_file_list,
        key=lambda x: int(re.search(r"normal_(\d+)\.lines\.txt", str(x)).group(1)),
    )
    return list(zip(image_list, canny_list, annotator_file_list))


def update_workflow(
    json_file_path: Path,
    seed,
    original_image_filename,
    canny_image_filename,
    positive_prompt,
    negative_prompt,
    output_image_name,
):
    json_data = load_json_file(json_file_path)
    if json_file_path == JSON_FILE_PATH_CANNY_P2P:
        json_data["12"]["inputs"]["image"] = original_image_filename
        json_data["38"]["inputs"]["image"] = canny_image_filename
        json_data["6"]["inputs"]["text"] = positive_prompt
        json_data["7"]["inputs"]["text"] = negative_prompt
        json_data["3"]["inputs"]["seed"] = seed
        json_data["37"]["inputs"]["filename_prefix"] = output_image_name
    elif json_file_path == JSON_FILE_PATH_CANNY:
        json_data["12"]["inputs"]["image"] = canny_image_filename
        json_data["6"]["inputs"]["text"] = positive_prompt
        json_data["7"]["inputs"]["text"] = negative_prompt
        json_data["3"]["inputs"]["seed"] = seed
        json_data["37"]["inputs"]["filename_prefix"] = output_image_name
    else:
        raise ValueError(f"Unsupported workflow: {json_file_path}")
    return json_data


def wait_for_server(timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{COMFYUI_API_URL}/system_stats", timeout=10)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("ComfyUI server did not become ready in time")


def wait_for_output(target_path: Path, timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        if target_path.exists() and target_path.stat().st_size > 0:
            return
        time.sleep(2)
    raise TimeoutError(f"Timeout waiting for output: {target_path}")


def process_single_label(
    label, settings, original_image_filename, canny_image_filename, image_idx
):
    dest = DATA_ROOT / label / f"{label}_{image_idx}.jpg"
    if dest.exists():
        print(f"skip {dest.name}")
        return dest
    output_prefix = dest.name
    workflow = update_workflow(
        settings["workflow"],
        settings["seed"],
        original_image_filename,
        canny_image_filename,
        settings["positive_prompt"],
        settings["negative_prompt"],
        output_prefix,
    )
    submit_prompt(workflow)
    output_png = OUTPUT_ROOT / f"{output_prefix}_00001_.png"
    wait_for_output(output_png)
    shutil.move(str(output_png), str(dest))
    print(f"done {dest.name}")
    return dest


def parse_slice(total):
    start_idx = int(os.environ.get("HGLANE_START_IDX", "0"))
    end_idx_raw = os.environ.get("HGLANE_END_IDX")
    limit_raw = os.environ.get("HGLANE_LIMIT")
    end_idx = total if end_idx_raw is None else min(total, int(end_idx_raw))
    if limit_raw is not None:
        end_idx = min(end_idx, start_idx + int(limit_raw))
    return start_idx, end_idx


def main():
    ensure_dirs()
    wait_for_server()
    items = build_work_items()
    start_idx, end_idx = parse_slice(len(items))
    detector = CannyDetector()
    print(f"total_items={len(items)} start={start_idx} end={end_idx}")
    for image_path, canny_path, annotator_path in items[start_idx:end_idx]:
        image_idx = re.search(r"normal_(\d+)\.jpg", image_path.name).group(1)
        input_image = cv2.imread(str(image_path))
        if input_image is None:
            print(f"Failed to load image: {image_path}")
            continue
        if not canny_path.exists():
            detected_map = detector(
                input_image, str(annotator_path), LOW_THRESHOLD, HIGH_THRESHOLD
            )
            cv2.imwrite(str(canny_path), detected_map)
            print(f"canny {canny_path.name}")
        original_image_filename = upload_image(image_path)
        canny_image_filename = upload_image(canny_path)
        for label, settings in LABEL_SETTINGS.items():
            process_single_label(
                label,
                settings,
                original_image_filename,
                canny_image_filename,
                image_idx,
            )


if __name__ == "__main__":
    main()
