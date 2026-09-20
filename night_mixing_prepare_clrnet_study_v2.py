from pathlib import Path
import json
import os
import random

import cv2


STUDY_ROOT = Path("/root/autodl-tmp/night_mixing_study_v2")
REAL_ROOT = Path("/root/autodl-tmp/CULane")
SYN_ROOT = Path("/root/autodl-tmp/HG-Lane/CULane_6x5k")

SEED = 20260617
DAY_THR = 80.0
NIGHT_THR = 60.0
REAL_NIGHT_FULL = 3500
REAL_BUDGETS = [100, 500, 1000]
MIX_RATIOS = [
    ("mix_syn100_real0", 3500, 0),
    ("mix_syn75_real25", 2625, 875),
    ("mix_syn50_real50", 1750, 1750),
    ("mix_syn25_real75", 875, 2625),
    ("mix_syn0_real100", 0, 3500),
]


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def read_lines(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_lines(path, lines):
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n")


def symlink_force(target, link_path):
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            raise RuntimeError(f"Refusing to replace real directory: {link_path}")
        link_path.unlink()
    os.symlink(target, link_path)


def img_rel_from_gt(line):
    return line.split()[0].lstrip("/")


def relabel_real_train_gt(line):
    parts = line.split()
    img, mask, *rest = parts
    return " ".join([f"/real_culane{img}", f"/real_culane{mask}", *rest])


def relabel_syn_train_gt(line):
    parts = line.split()
    img, mask, *rest = parts
    return " ".join([f"/syn_hglane{img}", f"/syn_hglane{mask}", *rest])


def real_eval(line):
    return f"/real_culane{line if line.startswith('/') else '/' + line}"


def syn_eval(line):
    return f"/syn_hglane{line if line.startswith('/') else '/' + line}"


def mean_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read {path}")
    return float(img.mean())


def build_configs(experiments):
    config_root = STUDY_ROOT / "configs"
    for exp in experiments:
        cfg_path = config_root / f"{exp['name']}.py"
        work_dir = STUDY_ROOT / "runs" / exp["name"]
        dataset_root = STUDY_ROOT / "datasets" / exp["name"]
        cfg_text = f"""net = dict(type='Detector', )

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet18',
    pretrained=False,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)

heads = dict(type='CLRHead',
             num_priors=192,
             refine_layers=3,
             fc_hidden_dim=64,
             sample_points=36)

iou_loss_weight = 2.
cls_loss_weight = 2.
xyt_loss_weight = 0.2
seg_loss_weight = 1.0

work_dirs = "{work_dir}"

neck = dict(type='FPN',
            in_channels=[128, 256, 512],
            out_channels=64,
            num_outs=3,
            attention=False)

test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

epochs = 12
batch_size = 24

optimizer = dict(type='AdamW', lr=0.6e-3)
total_iter = ({exp['train_count']} // batch_size) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_ep = 12
save_ep = 12

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = '{dataset_root}'
dataset_type = 'CULane'
dataset = dict(train=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='train',
    processes=train_process,
),
val=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='val',
    processes=val_process,
),
test=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
))

workers = 4
log_interval = 100
num_classes = 4 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
"""
        cfg_path.write_text(cfg_text)


def main():
    rng = random.Random(SEED)
    ensure_dir(STUDY_ROOT)
    ensure_dir(STUDY_ROOT / "datasets")
    ensure_dir(STUDY_ROOT / "configs")
    ensure_dir(STUDY_ROOT / "results")
    ensure_dir(STUDY_ROOT / "logs")
    ensure_dir(STUDY_ROOT / "scripts")

    real_train_raw = read_lines(REAL_ROOT / "list" / "train_gt.txt")
    cache_path = STUDY_ROOT / "results" / "train_brightness_cache.json"
    if cache_path.exists():
        brightness = json.loads(cache_path.read_text())
    else:
        brightness = {}
    missing = [line for line in real_train_raw if line not in brightness]
    for idx, line in enumerate(missing, 1):
        brightness[line] = mean_gray(REAL_ROOT / img_rel_from_gt(line))
        if idx % 5000 == 0:
            cache_path.write_text(json.dumps(brightness))
    cache_path.write_text(json.dumps(brightness))

    real_day_raw = [line for line in real_train_raw if brightness[line] >= DAY_THR]
    real_night_raw = [line for line in real_train_raw if brightness[line] < NIGHT_THR]
    ambiguous_raw = [line for line in real_train_raw if NIGHT_THR <= brightness[line] < DAY_THR]

    rng.shuffle(real_day_raw)
    rng.shuffle(real_night_raw)

    real_day_train = [relabel_real_train_gt(x) for x in real_day_raw]
    real_night_full = [relabel_real_train_gt(x) for x in real_night_raw[:REAL_NIGHT_FULL]]

    syn_train_gt_all = [relabel_syn_train_gt(x) for x in read_lines(SYN_ROOT / "list" / "train_gt.txt")]
    syn_night_train = [x for x in syn_train_gt_all if x.split()[0].startswith("/syn_hglane/night/")]

    real_val = [real_eval(x) for x in read_lines(REAL_ROOT / "list" / "val.txt")]
    real_night_test = [real_eval(x) for x in read_lines(REAL_ROOT / "list" / "test_split" / "test8_night.txt")]
    syn_night_test = [syn_eval(x) for x in read_lines(SYN_ROOT / "list" / "test.txt") if x.startswith("/night/")]

    write_lines(STUDY_ROOT / "results" / "real_day_pool.txt", [x.split()[0] for x in real_day_raw])
    write_lines(STUDY_ROOT / "results" / "real_night_pool.txt", [x.split()[0] for x in real_night_raw])
    write_lines(STUDY_ROOT / "results" / "ambiguous_pool.txt", [x.split()[0] for x in ambiguous_raw])
    write_lines(STUDY_ROOT / "results" / "real_night_test.txt", real_night_test)
    write_lines(STUDY_ROOT / "results" / "syn_night_test.txt", syn_night_test)

    experiments = []

    def add_exp(name, extra_syn, extra_real):
        train_lines = list(real_day_train)
        if extra_syn:
            train_lines.extend(syn_night_train[:extra_syn])
        if extra_real:
            train_lines.extend(real_night_full[:extra_real])
        exp_root = STUDY_ROOT / "datasets" / name
        ensure_dir(exp_root / "list")
        symlink_force(REAL_ROOT, exp_root / "real_culane")
        symlink_force(SYN_ROOT, exp_root / "syn_hglane")
        write_lines(exp_root / "list" / "train_gt.txt", train_lines)
        write_lines(exp_root / "list" / "val.txt", real_val)
        write_lines(exp_root / "list" / "test.txt", real_night_test + syn_night_test)
        write_lines(exp_root / "list" / "test_real_night.txt", real_night_test)
        write_lines(exp_root / "list" / "test_syn_night.txt", syn_night_test)
        meta = {
            "name": name,
            "extra_syn_night": extra_syn,
            "extra_real_night": extra_real,
            "train_count": len(train_lines),
        }
        (exp_root / "meta.json").write_text(json.dumps(meta, indent=2))
        experiments.append(meta)

    add_exp("exp1_day_only", 0, 0)
    add_exp("exp1_day_plus_syn_night_full", len(syn_night_train), 0)
    add_exp("exp1_day_plus_real_night_full", 0, REAL_NIGHT_FULL)
    for name, syn_count, real_count in MIX_RATIOS:
        add_exp(f"exp3_{name}", syn_count, real_count)
    for budget in REAL_BUDGETS:
        add_exp(f"exp4_real_budget_{budget}", 0, budget)
        add_exp(f"exp4_real_budget_{budget}_plus_syn_full", len(syn_night_train), budget)

    build_configs(experiments)
    (STUDY_ROOT / "results" / "experiments.json").write_text(json.dumps(experiments, indent=2))

    summary = {
        "seed": SEED,
        "day_threshold": DAY_THR,
        "night_threshold": NIGHT_THR,
        "real_train_total": len(real_train_raw),
        "real_day_count": len(real_day_train),
        "real_night_available": len(real_night_raw),
        "real_ambiguous_count": len(ambiguous_raw),
        "real_night_full_used": REAL_NIGHT_FULL,
        "syn_night_train_count": len(syn_night_train),
        "real_night_test_count": len(real_night_test),
        "syn_night_test_count": len(syn_night_test),
        "experiments": [x["name"] for x in experiments],
    }
    (STUDY_ROOT / "results" / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
