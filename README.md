# scPilot

# Running Guide
## set-up

### 1. API key
replace the OPENAI_API_KEY and GOOGLE_API_KEY in /config/settings.py with your own <be>.

### 2. env install
refer to requirements.txt for python environment. Note that you will need numpy < 2.0 to run py-Monocle.

### 3. large file download

you should download the files in our anonymous google drive:
https://drive.google.com/drive/folders/18AFRwp0eEftBgy2_WfQBtXrp39z4yn2w?usp=sharing

you should then put it into the scPilot folder and name it as uploads/

## running 3 tasks

### 0. LLM usage
currently we support 2 LLM API sources: openai and google. You should define it using the 'model_provider'. Then, choose a 'model_name' that exists, such as "gpt-4o".

### 1. Cell type annotation
run Task1_scPilot.py for scPilot version, run Task1_direct.py for direct prompting, and use Task1_scoring for scoring annotation.
Task 1 results are in Task1_results folder.

For each of the python files, please refer to its config section to specify the input / output folders.  

You should refer to CellTypeAnnotationDatasets.xlsx to fill in the configs related to dataset, such as original_grouping, correct_column, output_column, species and initial_hypothesis.

You can also change the LLM usage in the config section.

### 2. Trajectory inference
for 3 datasets used in trajectory inference task, run Traj_scPilot_1 to 3.ipynb for scPilot trajectory inference.
run Traj_direct_1 to 3.ipynb for direct trajectory inference.
Task 2 results are in Task2_results folder.

You can specify the LLM usage in the first notebook cell. Other than that, you can let itself run. In direct version notebooks, you may need to manually copy the tree from LLM response to ensure smooth running.

### 3. GRN TF-gene prediction
run Task3_combined.py, it will generate direct version and scPilot version prediction for the same questions in one run. 
Task 3 results are in Task3_results folder.

You can change the LLM usage in the config section.
If you want to try different tissues, you can change PREDICT_CONTEXT to "Liver" or "Stomach" or "Kidney".


# Current results
We have put all results (including additional experiment, ablation study, etc) in the rebuttal stage, in to https://drive.google.com/file/d/1JmDV2zEK1Rw3QTFJP7WV39Ptp37KBpQB/view?usp=sharing. 




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

[cite_start]Unlike traditional "tool agents" that simply write code, scPilot performs **Omics-Native Reasoning**: it directly inspects data summaries, explicitly articulates biological hypotheses, and iteratively refines its conclusions using on-demand bioinformatics tools[cite: 1335, 1367].

### Key Features
- [cite_start]**🔬 Biological Context First:** Incorporates tissue, species, and experimental metadata into reasoning[cite: 1480].
- [cite_start]**🔄 Iterative Refinement:** Self-corrects hypotheses based on computational evidence (e.g., dotplots, marker genes)[cite: 1481].
- [cite_start]**📊 Transparent & Auditable:** Generates full reasoning traces, not just black-box vectors[cite: 1368].

---

## 🚀 Performance Highlights
[cite_start]Evaluated on **scBench** (9 expertly curated datasets)[cite: 1346], scPilot demonstrates superior performance over direct LLM prompting and traditional pipelines.

| Task | Metric Improvement | Key Result |
| :--- | :--- | :--- |
| **Cell-Type Annotation** | **+11% Accuracy** | [cite_start]Iterative reasoning lifts average accuracy by 11% compared to one-shot methods. |
| **Trajectory Inference** | **-30% Graph Error** | [cite_start]Cuts trajectory graph-edit distance by 30% (using Gemini-2.5-Pro). |
| **GRN Prediction** | **+0.03 AUROC** | [cite_start]Improves Gene Regulatory Network prediction AUROC over baseline baselines. |

### 💰 Cost Efficiency
scPilot is designed to be efficient. [cite_start]Average costs per run (using Gemini-2.5-Pro) are minimal:
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
Download large file dependencies from our [anonymous Google Drive](https://drive.google.com/drive/folders/18AFRwp0eEftBgy2_WfQBtXrp39z4yn2w?usp=sharing) and place them in the `scPilot/uploads/` folder.

---

## 🏃 Running scPilot Tasks

### 0. Configure LLM
In your config, set `model_provider` (openai/google) and `model_name` (e.g., `gpt-4o`, `gemini-2.5-pro`).

### 1. Cell Type Annotation
* **Run:** `Task1_scPilot.py` (scPilot version) or `Task1_direct.py` (Direct prompting).
* **Config:** Update `CellTypeAnnotationDatasets.xlsx` with your dataset details (grouping, species, etc.).
* **Output:** See `Task1_results/`.

### 2. Trajectory Inference
* **Run:** `Traj_scPilot_1.ipynb` through `3.ipynb`.
* **Note:** Direct version notebooks may require manual tree copying.
* **Output:** See `Task2_results/`.

### 3. GRN TF-Gene Prediction
* **Run:** `Task3_combined.py` (Generates both direct and scPilot predictions).
* **Options:** Change `PREDICT_CONTEXT` to "Liver", "Stomach", or "Kidney" in the config.
* **Output:** See `Task3_results/`.

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
