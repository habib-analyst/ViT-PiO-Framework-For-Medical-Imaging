# Medical AI Models - ViT + PiO

Comprehensive deep learning models for medical image classification using Vision Transformer (ViT) and Perceiver IO architecture.

## ✨ Author Information

**Author:** Ayesha Khaliq, Fahad Ahmad, Habib Ur Rehman, Saad Awadh Alanazi, Hamza Haleem, Kashaf Junaid, Elisavet Andrikopoulou

## ✅ Publication

This work has been published in Computational Biology and Chemistry (2025):

Habib Ur Rehan et al., "A hybrid Vision Transformer + Perceiver IO framework for multi-disease medical imaging classification", Computational Biology and Chemistry, 2025. DOI: https://doi.org/10.1016/j.compbiolchem.2025.108586

The repository contains the code and experiments corresponding to the paper.

## 📋 Project Overview

This project contains three AI models trained to detect and classify medical conditions from image data:

1. **Brain Model** - Alzheimer's vs Stroke Classification
2. **Lung Model** - Lung Cancer vs Pneumonia Classification
3. **Skin Model** - Skin Disease Classification (Melanoma, Tinea)

## 🏗️ Architecture

All models use a hybrid approach combining:
- **Vision Transformer (ViT)** - For patch-based image understanding
- **Perceiver IO** - For efficient cross-attention mechanisms
- **Data Augmentation** - Random flip, rotation, zoom, and contrast adjustments

## 📁 Project Structure

```
Research-Project-Medical-Models/
├── Brain Model/
│   ├── Brain Model/
│   │   ├── neuro_code.py          # Brain model training script
│   │   ├── CSVF.py                # CSV processing utility
│   │   ├── EM2.py                 # Data preprocessing
│   │   ├── FCS1.py                # Feature processing
│   │   ├── NPYF.py                # NumPy file utilities
│   │   ├── pi.py                  # Preprocessing interface
│   │   └── stroke_alzheimer_dataset.csv
│   ├── AlzVsStr/                  # Dataset folders
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── Model/                     # Trained models
│   ├── NPYF/                      # Preprocessed NumPy arrays
│   └── Predicted_Skin/
├── Lung Model/
│   ├── Lung Model/
│   │   ├── lung_code.py           # Lung model training script
│   │   ├── CSVF.py
│   │   ├── EM2.py
│   │   ├── FCS1.py
│   │   ├── NPYF.py
│   │   ├── pi.py
│   │   └── Pneumonia_dataset.csv
│   ├── CancerVsPneumonia/         # Dataset folders
│   ├── Model/                     # Trained models
│   └── NPYF/
├── Skin Model/
│   ├── Skin Model/
│   │   ├── skin_code.py           # Skin model training script
│   │   ├── CSVF.py
│   │   ├── EM2.py
│   │   ├── FCS1.py
│   │   ├── NPYF.py
│   │   ├── pi.py
│   │   └── skin_disease_dataset.csv
│   ├── Skin Disease/              # Dataset folders
│   ├── Model/                     # Trained models
│   └── NPYFile/
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```

### Running the Models

Each model can be trained independently:

**Brain Model:**
```bash
cd "Brain Model/Brain Model"
python neuro_code.py
```

**Lung Model:**
```bash
cd "Lung Model/Lung Model"
python lung_code.py
```

**Skin Model:**
```bash
cd "Skin Model/Skin Model"
python skin_code.py
```

## 📊 Model Performance

All models include:
- **Early Stopping** - Prevents overfitting
- **Learning Rate Reduction** - Adapts learning during training
- **Dropout Regularization** - 50% dropout in classification head
- **Data Validation** - Separate validation and test sets

## 🔧 Configuration

Models are configured with:
- **Input Shape**: 224×224×3 (RGB images)
- **Patch Size**: 16×16 pixels
- **Transformer Heads**: 4
- **Perceiver IO Blocks**: 3
- **Batch Size**: 32
- **Learning Rate**: 1e-4 (with adaptive reduction)
- **Epochs**: 5+ (with early stopping)

## 📈 Training Pipeline

1. **Data Loading** - Load NumPy arrays from NPYF/NPYFile directories
2. **Preprocessing** - Expand dims, normalize to [0,1], apply augmentation
3. **Model Building** - ViT encoder + Perceiver IO blocks
4. **Training** - With callbacks for early stopping and LR reduction
5. **Evaluation** - Test accuracy and loss metrics
6. **Visualization** - Learning curves and training history plots

## 💾 Model Outputs

Trained models are saved as HDF5 files:
- `Brain Model/Model/vit_perceiver_io_model.h5`
- `Lung Model/Model/vit_perceiver_io_model.h5`
- `Skin Model/Model/skin_disease_vit_perceiver_io_model.h5`

Classification reports are saved as:
- `*_classification_report.txt`

## 🌟 Highlights

- Proposes a hybrid AI using Vision Transformers and Perceiver IO for multi-disease medical image classification.
- Achieves high accuracy and low false positives across neurology, dermatology, and pulmonology domains.
- Reaches up to 1.00 recall for six diseases across benchmark datasets in neurology, skin, and lung conditions.
- Integrates a real-time chatbot for diagnostic image uploads with automated interpretation and confidence scores.
- First use of ViT + Perceiver IO for these diseases, surpassing CNN models in accuracy and computational efficiency.

## 🎯 Key Features

✅ Portable code using relative paths  
✅ Modular preprocessing utilities  
✅ Comprehensive data augmentation  
✅ Hybrid ViT + Perceiver IO architecture  
✅ Early stopping and learning rate scheduling  
✅ Detailed evaluation metrics  
✅ Learning curve visualization  

## 📝 Dataset Format

**Research Datasets:** https://drive.google.com/file/d/1hdE8f-aKgy0_jxF58chN6ovdjwa5ZdXO/view?usp=sharing

Datasets are organized as:
- **Training**: `*/train/Class1/` and `*/train/Class2/`
- **Validation**: `*/validation/Class1/` and `*/validation/Class2/`
- **Testing**: `*/test/Class1/` and `*/test/Class2/`

Preprocessed data is stored as NumPy arrays:
- `X_train.npy`, `y_train.npy`
- `X_validation.npy`, `y_validation.npy`
- `X_test.npy`, `y_test.npy`

## 🔍 Utility Scripts

- **CSVF.py** - CSV file processing and dataset management
- **EM2.py** - Data preprocessing and feature engineering
- **FCS1.py** - Feature computation and scaling
- **NPYF.py** - NumPy array file I/O operations
- **pi.py** - Preprocessing interface and pipeline

## 📧 Notes

- All paths use relative directories for cross-platform compatibility
- Models require GPU for efficient training (TensorFlow with CUDA)
- CPU training is supported but significantly slower
- Each model training takes approximately 10-20 minutes per epoch

## 📄 License

This project is for educational and research purposes.

---
