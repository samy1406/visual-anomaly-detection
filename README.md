# 🔍 Visual Anomaly Detection

> Detecting defects in industrial images using unsupervised deep learning — PatchCore & AutoEncoder on MVTec AD dataset.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green?logo=scikit-learn) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 🧩 Problem Statement 

Traditional quality control in manufacturing relies on manual inspection slow, expensive, and inconsistent. This project trains a model **only on normal (defect-free) images** and detects anomalies at inference time, without any labeled defect data.

**Key challenge:** No defect labels during training. The model must learn "what normal looks like" and flag deviations.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   MVTec AD Dataset                   │
│         (train: normal only | test: normal+defect)   │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────┴────────────┐
         ▼                          ▼
  ┌─────────────┐          ┌─────────────────┐
  │  PatchCore  │          │   AutoEncoder   │
  │  Approach   │          │    Approach     │
  └──────┬──────┘          └────────┬────────┘
         │                          │
  ResNet backbone            Encoder-Decoder
  → patch features           → reconstruction
  → FAISS memory bank        → pixel diff map
  → k-NN anomaly score       → anomaly score
         │                          │
         └──────────┬───────────────┘
                    ▼
          ┌──────────────────┐
          │  Anomaly Score   │
          │  + Heatmap       │
          │  + AUROC Eval    │
          └──────────────────┘
```

---

## 📁 Repository Structure

```
visual-anomaly-detection/
│
├── src/
│   ├── datasets/
│   │   └── mvtec.py              # Dataset loader + transforms
│   ├── models/
│   │   ├── patchcore.py          # PatchCore implementation
│   │   └── autoencoder.py        # Convolutional AutoEncoder
│   ├── utils/
│   │   ├── metrics.py            # AUROC, threshold selection
│   │   └── visualization.py      # Heatmap overlay utils
│   └── train.py                  # Training entry point
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_patchcore_experiment.ipynb
│   └── 03_autoencoder_experiment.ipynb
│
├── data/
│   └── README.md                 # MVTec download instructions
│
├── demo/
│   └── sample_results.gif        # Heatmap overlay demo
│
├── requirements.txt
├── Dockerfile
└── LEARNING.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Download MVTec AD Dataset
```bash
# From https://www.mvtec.com/company/research/datasets/mvtec-ad
# Place under data/mvtec/
# Structure: data/mvtec/<category>/train/good/*.png
#                                  test/good/*.png
#                                  test/<defect_type>/*.png
```

### Run PatchCore
```bash
python src/train.py --method patchcore --category bottle --backbone resnet18
```

### Run AutoEncoder
```bash
python src/train.py --method autoencoder --category bottle --epochs 50
```

---

## 📊 Results

| Method | Category | AUROC | Inference Time |
|--------|----------|-------|---------------|
| PatchCore (ResNet18) | Bottle | ~97% | ~15ms/img |
| PatchCore (WideResNet50) | Bottle | ~99% | ~40ms/img |
| AutoEncoder (CNN) | Bottle | ~85% | ~5ms/img |

> Results vary by category. Textured surfaces (carpet, tile) are harder than object surfaces.

---

## 🧠 Key Concepts Explored

- **Unsupervised anomaly detection** — train on normal, test for deviation
- **PatchCore** — memory bank of normal patch features + k-NN at inference
- **Reconstruction-based detection** — AE fails to reconstruct anomalies well
- **AUROC** — threshold-free metric ideal for anomaly detection evaluation
- **Anomaly heatmaps** — pixel-level localization, not just image-level classification

---

## 📦 Requirements

```
torch>=2.0
torchvision>=0.15
scikit-learn>=1.2
faiss-cpu>=1.7
numpy
opencv-python
matplotlib
tqdm
```

---

## 🔗 References

- [MVTec AD Dataset Paper (Bergmann et al., 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf)
- [PatchCore Paper (Roth et al., 2022)](https://arxiv.org/abs/2106.08265)
- [Anomaly Detection Benchmark](https://github.com/openvinotoolkit/anomalib)

---

## 💡 What I Learned

See [LEARNING.md](./LEARNING.md) for detailed notes on concepts, implementation challenges, and interview talking points.
