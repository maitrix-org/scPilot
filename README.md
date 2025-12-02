<div align="center">

<img src="https://path-to-your-figure-1-or-2.png" alt="scPilot Framework" width="800"/>

# scPilot: Large Language Model Reasoning for Automated Single-Cell Analysis

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue)](https://github.com/maitrix-org/scPilot)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**The first systematic framework for omics-native reasoning.**

</div>

## 📖 Overview
**scPilot** operates as a true scientific assistant that automates core single-cell analyses—**Cell-Type Annotation**, **Trajectory Inference**, and **GRN Prediction**. 

Unlike traditional "tool agents" that simply write code, scPilot performs **Omics-Native Reasoning**: it directly inspects data summaries, explicitly articulates biological hypotheses, and iteratively refines its conclusions using on-demand bioinformatics tools.

### Key Features
- **🔬 Biological Context First:** Incorporates tissue, species, and experimental metadata into reasoning.
- **🔄 Iterative Refinement:** Self-corrects hypotheses based on computational evidence (e.g., dotplots, marker genes).
- **📊 Transparent & Auditable:** Generates full reasoning traces, not just black-box vectors.

---

## 🚀 Performance Highlights
Evaluated on **scBench** (9 expertly curated datasets), scPilot demonstrates superior performance over direct LLM prompting and traditional pipelines.

| Task | Metric Improvement | Key Result |
| :--- | :--- | :--- |
| **Cell-Type Annotation** | **+11% Accuracy** | Iterative reasoning lifts average accuracy by 11% compared to one-shot methods. |
| **Trajectory Inference** | **-30% Graph Error** | Cuts trajectory graph-edit distance by 30% (using Gemini-2.5-Pro). |
| **GRN Prediction** | **+0.03 AUROC** | Improves Gene Regulatory Network prediction AUROC over baseline baselines. |

### 💰 Cost Efficiency
scPilot is designed to be efficient. Average costs per run (using Gemini-2.5-Pro) are minimal:
* **Cell-type annotation (Retina):** ~$0.03 / run
* **Trajectory inference (Neocortex):** ~$0.04 / run
* **GRN TF-gene prediction:** ~$0.12 / run

---

## 🛠️ Installation & Setup

### 1. API Key Configuration
Replace `OPENAI_API_KEY` and `GOOGLE_API_KEY` in `/config/settings.py`.

### 2. Environment Setup
Refer to `requirements.txt`. 
> **Note:** `numpy < 2.0` is required for `py-Monocle`.

### 3. Data Preparation
Download large file dependencies from [Google Drive](https://drive.google.com/drive/folders/18AFRwp0eEftBgy2_WfQBtXrp39z4yn2w?usp=sharing) and place them in the `scPilot/uploads/` folder.

---

## 🏃 Running scPilot Tasks

### 0. Configure LLM
In your config, set `model_provider` (openai/google) and `model_name` (e.g., `gpt-4o`, `gemini-2.5-pro`).

### 1. Cell Type Annotation
* **Run:** `Task1_scPilot.py` (scPilot version) or `Task1_direct.py` (Direct prompting).
* **Config:** Update `CellTypeAnnotationDatasets.xlsx` with your dataset details (grouping, species, etc.).


### 2. Trajectory Inference
* **Run:** `Traj_scPilot_1.ipynb` through `3.ipynb` for scPilot; `Traj_Direct_1.ipynb` through `3.ipynb` for Direct prompting.
* **Note:** Direct version notebooks may require manual tree copying.

### 3. GRN TF-Gene Prediction
* **Run:** `Task3_combined.py` (Generates both direct and scPilot predictions).
* **Options:** Change `PREDICT_CONTEXT` to "Liver", "Stomach", or "Kidney" in the config.

### Current Results

Results used in the paper can be found in the [google drive](https://drive.google.com/file/d/1JmDV2zEK1Rw3QTFJP7WV39Ptp37KBpQB/view?usp=sharing).

This result includes 3 tasks (`Task1_results/`, `Task2_results/` and `Task3_results/`), along with additional experiment, ablation study, etc) in the rebuttal stage. 

---

## 🔗 Citation
If you use scPilot, please cite our NeurIPS 2025 paper:

```bibtex
@inproceedings{
gao2025scpilot,
title={scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery},
author={Yiming Gao and Zhen Wang and Jefferson Chen and Mark Antkowiak and Mengzhou Hu and JungHo Kong and Dexter Pratt and Jieyuan Liu and Enze Ma and Zhiting Hu and Eric P. Xing},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=Vzi96rTe4w}
}
