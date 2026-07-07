# NightLaneSynth

Official repository for **Synthetic Night Data for Nighttime Lane Detection: Efficiency Analysis and Data Mixing Strategies**.

NightLaneSynth investigates how synthetic nighttime data can improve lane detection under real nighttime driving scenarios. The study focuses on the utility of synthetic nighttime images generated from HG-Lane, their domain gap with real nighttime data, and effective synthetic-real data mixing strategies for reducing annotation costs.

## Overview

Nighttime lane detection is challenging due to low visibility, motion blur, sensor noise, and complex illumination. Collecting and annotating real nighttime lane data is expensive, while synthetic data provides a scalable alternative.

This project studies three core questions:

1. Can synthetic nighttime data improve a daytime-trained lane detector?
2. How large is the gap between synthetic and real nighttime data?
3. Can synthetic data reduce the need for annotated real nighttime samples?

## Key Findings

- Synthetic nighttime data improves real nighttime lane detection compared with daytime-only training.
- Real nighttime data still outperforms purely synthetic nighttime data, showing a clear synthetic-to-real domain gap.
- Mixing synthetic and real nighttime data can outperform using either source alone.
- Synthetic data helps reduce real nighttime annotation requirements while maintaining competitive performance.

## Method

The experiments use:

- **HG-Lane** as the source of synthetic nighttime images.
- **CULane** as the source of real-world lane images.
- **CLRNet** as the lane detection baseline.
- Real nighttime evaluation is conducted on the nighttime subset of CULane.

The study evaluates three settings:

### 1. Synthetic Night Supplement

Compares:

- Daytime-only training
- Daytime + synthetic nighttime training
- Daytime + real nighttime training

### 2. Mixing Ratio Analysis

Studies different synthetic-real nighttime ratios, including:

- 100% synthetic / 0% real
- 75% synthetic / 25% real
- 50% synthetic / 50% real
- 25% synthetic / 75% real
- 0% synthetic / 100% real

### 3. Annotation Budget Analysis

Evaluates whether synthetic nighttime data can compensate for limited real nighttime annotations under different real-data budgets.

## Results Summary

### Synthetic Night Supplement

| Setting | Real F1@50 | Real F1@75 | Real mF1 | Syn. mF1 |
|---|---:|---:|---:|---:|
| Day only | 0.5853 | 0.4224 | 0.3808 | 0.6293 |
| Day + synthetic night | 0.6278 | 0.4496 | 0.4055 | 0.6905 |
| Day + real night | 0.7180 | 0.5150 | 0.4671 | 0.6564 |

### Synthetic-to-Real Gap

| Setting | Real F1@50 | Real F1@75 | Real mF1 | Syn. mF1 |
|---|---:|---:|---:|---:|
| Synthetic-only night | 0.6371 | 0.4437 | 0.4065 | 0.6879 |
| Real-only night | 0.7183 | 0.5115 | 0.4653 | 0.6510 |

### Best Mixing Strategy

The best real nighttime performance is achieved with a **25% synthetic / 75% real** nighttime data mixture:

| Mix Ratio (Syn/Real) | Real F1@50 | Real F1@75 | Real mF1 | Syn. mF1 |
|---|---:|---:|---:|---:|
| 25/75 | 0.7211 | 0.5192 | 0.4706 | 0.6788 |

## Repository Structure

NightLaneSynth/
├── configs/              # Training and evaluation configurations
├── data/                 # Dataset links or processed data lists
├── tools/                # Data preparation and evaluation scripts
├── experiments/          # Experiment scripts
├── results/              # Logs, checkpoints, and evaluation results
└── README.md

## Getting Started
###  1. Clone the repository

git clone https://github.com/jylEcho/NightLaneSynth.git
cd NightLaneSynth

###  2. Prepare datasets

Download the required datasets from their official sources:

HG-Lane: https://github.com/zdc233/HG-Lane
CULane: https://xingangpan.github.io/projects/CULane.html

Organize the datasets according to the expected project structure and update the dataset paths in the configuration files.

###  3. Train and evaluate

Use the provided configuration files to reproduce the synthetic supplement, mixing-ratio, and annotation-budget experiments.

# Example
python tools/train.py --config configs/nightlanesynth.yaml
python tools/eval.py --config configs/nightlanesynth.yaml

Please adjust script names and configuration paths according to the released implementation.

##  Data Availability
The data supporting this study are available in NightLaneSynth: https://github.com/jylEcho/NightLaneSynth.

NightLaneSynth is derived from publicly available resources:

HG-Lane: https://github.com/zdc233/HG-Lane
CULane: https://xingangpan.github.io/projects/CULane.html

Users should follow the licenses and usage terms of the original datasets.
