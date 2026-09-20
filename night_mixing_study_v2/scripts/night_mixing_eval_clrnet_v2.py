from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from clrnet.utils.config import Config
from clrnet.datasets import build_dataloader
from clrnet.models.registry import build_net
from clrnet.utils.culane_metric import eval_predictions


STUDY_ROOT = Path("/root/autodl-tmp/night_mixing_study_v2")


def find_latest_ckpt(exp_name: str) -> str:
    pattern = f"{STUDY_ROOT}/runs/{exp_name}/*/ckpt/11.pth"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    args = parser.parse_args()

    exp_name = args.exp_name
    cfg_path = STUDY_ROOT / "configs" / f"{exp_name}.py"
    data_root = STUDY_ROOT / "datasets" / exp_name
    out_root = STUDY_ROOT / "eval" / exp_name
    out_root.mkdir(parents=True, exist_ok=True)
    pred_dir = out_root / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(cfg_path))
    cfg.load_from = find_latest_ckpt(exp_name)
    cfg.gpus = 1
    if not hasattr(cfg, "seed"):
        cfg.seed = 0

    loader = build_dataloader(cfg.dataset.test, cfg, is_train=False)
    net = build_net(cfg)
    net = torch.nn.parallel.DataParallel(net, device_ids=range(cfg.gpus)).cuda()
    state = torch.load(cfg.load_from, map_location="cpu")
    net.load_state_dict(state["net"], strict=True)
    net.eval()

    predictions = []
    for batch in tqdm(loader, desc=f"Eval {exp_name}"):
        batch["img"] = batch["img"].cuda()
        with torch.no_grad():
            output = net(batch)
            lanes = net.module.heads.get_lanes(output)
            predictions.extend(lanes)

    loader.dataset.evaluate(predictions, str(pred_dir))

    real_test = data_root / "list" / "test_real_night.txt"
    syn_test = data_root / "list" / "test_syn_night.txt"
    res_real = eval_predictions(
        str(pred_dir),
        str(data_root),
        str(real_test),
        iou_thresholds=np.linspace(0.5, 0.95, 10),
        official=True,
    )
    res_syn = eval_predictions(
        str(pred_dir),
        str(data_root),
        str(syn_test),
        iou_thresholds=np.linspace(0.5, 0.95, 10),
        official=True,
    )

    out = {
        "exp_name": exp_name,
        "ckpt": cfg.load_from,
        "pred_dir": str(pred_dir),
        "real_night": {
            "F1@50": res_real[0.5]["F1"],
            "F1@75": res_real[0.75]["F1"],
            "mF1": res_real["mean"]["F1"],
        },
        "synthetic_night": {
            "F1@50": res_syn[0.5]["F1"],
            "F1@75": res_syn[0.75]["F1"],
            "mF1": res_syn["mean"]["F1"],
        },
    }
    out_path = STUDY_ROOT / "results" / f"{exp_name}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()