# ViT + Perceiver IO for Medical Imaging

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://github.com/habib-analyst/ViT-PiO-Framework-For-Medical-Imaging)
[![GitHub stars](https://img.shields.io/github/stars/habib-analyst/ViT-PiO-Framework-For-Medical-Imaging.svg)](https://github.com/habib-analyst/ViT-PiO-Framework-For-Medical-Imaging)

Hybrid Vision Transformer + Perceiver IO models for multi-disease medical image classification — three trained models covering brain, lung, and skin. Published in *Computational Biology and Chemistry* (Elsevier), 2025.

## Quickstart

```bash
git clone https://github.com/habib-analyst/ViT-PiO-Framework-For-Medical-Imaging
cd ViT-PiO-Framework-For-Medical-Imaging
pip install tensorflow keras numpy scikit-learn matplotlib
```

Each domain has its own training script with a modular preprocessing pipeline (`CSVF`, `EM2`, `FCS1`, `NPYF`, `pi`):

```bash
python "Brain Model/Brain Model/neuro_code.py"      # Alzheimer's vs stroke
python "Lung Model/Lung Model/lung_code.py"         # lung cancer vs pneumonia
python "Skin Model/Skin Model/skin_code.py"         # melanoma vs tinea
```

Datasets (Google Drive link in the original README) are organized as `*/train/`, `*/validation/`, `*/test/` with one folder per class; preprocessing converts them to `X_*.npy` / `y_*.npy` arrays. Trained models are saved as HDF5 (`*/Model/vit_perceiver_io_model.h5`); classification reports are written to `*_classification_report.txt`.

## Results (as reported in the paper)

| Domain | Task | Accuracy |
|---|---|---|
| Brain | Alzheimer's vs stroke | 0.99 |
| Lung | Lung cancer vs pneumonia | 0.98 |
| Skin | Melanoma vs tinea | 0.95 |

25,000 Kaggle images (10k neuro MRI, 10k dermoscopic, 5k chest X-ray/CT), 70/15/15 split. The paper also reports recall up to 1.00 for the six diseases. Note: GPU (TensorFlow/CUDA) is recommended — CPU training works but is substantially slower.

## How it works

1. **Vision Transformer (ViT)** — patch-based image understanding for long-range spatial dependencies.
2. **Perceiver IO** — cross-attention fusion over the ViT features, efficient on high-dimensional inputs.
3. **Training** — random flip/rotation/zoom/contrast augmentation, early stopping, learning-rate scheduling; evaluation with classification reports, confusion matrices, and learning-curve plots.

## Citation

```bibtex
@article{khaliq2025vitpio,
  author  = {Ayesha Khaliq and Fahad Ahmad and Habib Ur Rehman and Saad Awadh Alanazi and Hamza Haleem and Kashaf Junaid and Elisavet Andrikopoulou},
  title   = {Revolutionizing medical imaging: {A} cutting-edge {AI} framework with vision transformers and perceiver {IO} for multi-disease diagnosis},
  journal = {Computational Biology and Chemistry},
  volume  = {119},
  article = {108586},
  publisher = {Elsevier},
  year    = {2025},
  doi     = {10.1016/j.compbiolchem.2025.108586}
}
```

## License

For educational and research purposes. © 2026 Habib Ur Rehman

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
