<div align="center">

# 🌌 DMICF

### Dual-Perspective Disentangled Multi-Intent Alignment for Enhanced Collaborative Filtering

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.1.0-red?style=flat-square&logo=pytorch">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python">
  <img src="https://img.shields.io/badge/Platform-Ubuntu-green?style=flat-square&logo=ubuntu">
  <img src="https://img.shields.io/badge/GPU-RTX4090D-76B900?style=flat-square&logo=nvidia">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square">
</p>

<p align="center">
  Official PyTorch implementation of the paper:<br>
  <b>Dual-Perspective Disentangled Multi-Intent Alignment for Enhanced Collaborative Filtering</b>
</p>

</div>

---

# ✨ Overview

DMICF is a collaborative filtering framework that models user--item interactions through:

- 🔷 **Dual-perspective structural encoding**
- 🧠 **Prototype-aware variational multi-intent modeling**
- 🎯 **Interaction-driven semantic alignment**
- ⚡ **Late fusion interaction prediction**

Unlike conventional unified-space recommenders, DMICF explicitly preserves complementary user-centric and item-centric semantics prior to fusion, enabling more expressive and disentangled interaction modeling.

---

# 🚀 Key Features

## 🔁 Reproducibility-Oriented Release

For every dataset used in our experiments, we provide:

- ✅ Trained checkpoints for **every training epoch**
- ✅ Complete evaluation scripts
- ✅ Reproducible training configurations
- ✅ Full experimental settings

This facilitates:

- detailed training dynamics analysis,
- result verification,
- fair comparison and reproducibility.

---

## 🧩 Intent Disentanglement Analysis Toolkit

We additionally release:

- 📌 Complete implementation of **user interaction group partitioning**
- 📊 Quantitative disentanglement evaluation tools
- 🎨 Qualitative intent visualization utilities
- 🔍 Semantic alignment analysis scripts

These tools support comprehensive analysis of learned latent intents and interaction semantics.

---

## ⚡ Efficient High-Order Graph Construction

DMICF constructs high-order user--user and item--item homogeneous graphs following the strategy adopted in IPCCF.

To improve scalability on large-scale datasets, we further provide a **GPU-accelerated graph construction implementation**:

```text
ml10m/uu_graph.py
```
which significantly improves preprocessing efficiency compared to the original implementation.

---

# 🖥️ Environment

All experiments are conducted on:

- Ubuntu Server
- NVIDIA RTX 4090D GPU

The complete dependency configuration is provided in:

```text
environment.yml
```

Main dependencies include:

```python
torch==2.1.0
numpy==1.22.3
scipy==1.9.3
pandas==1.5.0
scikit-learn==1.1.2
```
# 📂 Datasets

We directly adopt the three representative recommendation datasets released by [**LightGCL**](https://github.com/HKUDS/LightGCL/tree/main/data), covering diverse interaction distributions and sparsity characteristics.

---

# ⚙️ Quick Start

## 🔧 Training

```bash
python train.py
```
## 📈 Evaluation

```bash
python eval_results.py
```
# 📝 Notes

- DMICF uses a unified experimental configuration across all datasets.
- Only minimal dataset-specific modifications are required during evaluation.
- All reported experimental results are averaged over five independent runs.

---

# 📌 Repository Structure

```text
DMICF/
├── train.py
├── eval_results.py
├── environment.yml
├── models/
├── datasets/
├── utils/
├── analysis/
├── checkpoints/
└── ml10m/
    └── uu_graph.py
```
# 📖 Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{DMICF,
  title={Dual-Perspective Disentangled Multi-Intent Alignment for Enhanced Collaborative Filtering},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
```
# 🙏 Acknowledgement

Part of the high-order graph construction pipeline is inspired by the implementation of IPCCF. We thank the authors for making their code publicly available.

---

<div align="center">

### ⭐ If you find this project useful, please consider giving it a star! ⭐

</div>
