# 🍎 Apple Leaf Disease Classification
### Supervised vs. Unsupervised Machine Learning — A Comparative Study

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

<br/>

**Classifying apple leaf diseases from raw pixel images using two different ML paradigms**

[📓 Random Forest Notebook](#-project-structure) · [📓 K-Means Notebook](#-project-structure) · [📊 Results](#-results) · [🚀 Quick Start](#-quick-start)

</div>

---

## 📌 Overview

This project tackles the real-world problem of **automated apple leaf disease detection** using machine learning on image data — no deep learning required. Two fundamentally different approaches are implemented and compared:

| Approach | Algorithm | Type | Accuracy |
|---|---|---|---|
| 🌲 Notebook 1 | Random Forest + GridSearchCV | **Supervised** | **75.00%** |
| 🔵 Notebook 2 | K-Means + PCA | **Unsupervised** | **24.17%** |

The project demonstrates **why labelled data matters**, how supervised and unsupervised methods behave on the same dataset, and what the performance gap tells us about feature representation in image classification.

---

## 🌿 Disease Classes

The model classifies apple leaf images into **4 categories**:

| Class | Description | Training Samples |
|---|---|---|
| 🟤 **Apple Scab** | Dark scabby lesions — *Venturia inaequalis* | 2,016 |
| ⚫ **Black Rot** | Concentric ring lesions — *Botryosphaeria obtusa* | 1,987 |
| 🟠 **Cedar Apple Rust** | Bright orange pustules — *Gymnosporangium* | 880 |
| 🟢 **Healthy** | Disease-free leaves | 1,316 |

> Dataset source: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) (Hughes & Salathé, 2015)

---

## 📊 Results

### 🌲 Random Forest — 75% Accuracy

```
Best Parameters: {'max_depth': 10, 'n_estimators': 100}
Test Accuracy:   75.00%

                        precision  recall  f1-score  support
      Apple Apple Scab    0.69     0.77     0.73      403
       Apple Black Rot    0.76     0.77     0.77      398
Apple Cedar Apple Rust    0.83     0.73     0.78      176
         Apple Healthy    0.80     0.70     0.74      263

              accuracy                       0.75     1240
           weighted avg   0.76     0.75     0.75     1240
```

### 🔵 K-Means — 24.17% Accuracy

```
Explained Variance (50 PCA components): 80%
K-Means Clustering Accuracy:            24.17%

(Near-random performance — confirms raw pixels are insufficient
 for unsupervised disease separation)
```

---

## 🗂 Project Structure

```
apple-leaf-disease-classification/
│
├── 📓 Project_Random_Forest.ipynb     # Supervised approach — Random Forest
├── 📓 Project_K_Means.ipynb           # Unsupervised approach — K-Means + PCA
│
├── data/
│   └── dataset_split/
│       ├── train/
│       │   ├── Apple Apple scab/
│       │   ├── Apple Black rot/
│       │   ├── Apple Cedar apple rust/
│       │   └── Apple healthy/
│       └── test/
│
├── Apple_Disease_Report.pdf           # Full comparative research report
└── README.md
```

---

## ⚙️ Methodology

### 🌲 Approach 1 — Random Forest (Supervised)

```
Image → Resize (64×64) → BGR→RGB → Flatten (12,288D)
      → LabelEncode → Train/Test Split (80/20, stratified)
      → GridSearchCV (5-fold CV) → Random Forest
      → Evaluate → Predict
```

**Key Steps:**
- Images resized to `64×64` and flattened into 12,288-dimensional vectors
- Labels encoded numerically using `LabelEncoder`
- Stratified 80/20 train-test split
- `GridSearchCV` tunes `n_estimators` ∈ {50, 100} and `max_depth` ∈ {None, 10} with 5-fold CV
- Best model evaluated on held-out test set

### 🔵 Approach 2 — K-Means + PCA (Unsupervised)

```
Image → Resize (32×32) → BGR→RGB → Flatten (3,072D)
      → PCA (50 components, 80% variance) → K-Means (k=4)
      → Majority Vote Label Mapping → Evaluate → Predict
```

**Key Steps:**
- Images resized to `32×32` for computational efficiency
- PCA reduces 3,072D → 50D (retaining 80% of variance)
- K-Means clusters with `n_clusters=4`, `n_init=10`
- Each cluster mapped to its majority true label via `scipy.stats.mode`

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/apple-leaf-disease-classification.git
cd apple-leaf-disease-classification
```

### 2. Install Dependencies

```bash
pip install numpy opencv-python matplotlib seaborn pandas scikit-learn scipy jupyter
```

### 3. Download the Dataset

Download the PlantVillage Apple subset from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) and place it as:

```
data/dataset_split/train/<class_name>/
```

### 4. Update the Path

In both notebooks, update the `path` variable to your local dataset location:

```python
# Windows
path = r"C:\Users\YourName\Desktop\apple-leaf-disease-classification\data\dataset_split\train"

# Mac / Linux
path = "/home/yourname/datasets/dataset_split/train"
```

### 5. Run the Notebooks

```bash
jupyter notebook
```

Open `Project_Random_Forest.ipynb` or `Project_K_Means.ipynb` and run all cells.

---

## 📦 Requirements

```txt
numpy
opencv-python
matplotlib
seaborn
pandas
scikit-learn
scipy
jupyter
```

Install all at once:

```bash
pip install numpy opencv-python matplotlib seaborn pandas scikit-learn scipy jupyter
```

> ⚠️ **Performance Tip:** If your dataset is on OneDrive or a network drive, copy it to your local disk first. Reading thousands of images from cloud-synced storage can be 10× slower.

---

## 🔍 Key Findings

- **Random Forest (75%)** significantly outperforms **K-Means (24.17%)** because it learns from labelled data
- **Cedar Apple Rust** has the highest precision (0.83) — visually most distinct class
- **K-Means** collapses all clusters onto one dominant class in pixel space — raw pixels lack semantic separation
- **PCA** with 50 components captures 80% of image variance but is insufficient for unsupervised class separation
- The **50.83% accuracy gap** quantifies the value of annotation in image classification pipelines

---

## 🔮 Future Improvements

- [ ] **CNN Feature Extraction** — Replace raw pixels with ResNet/VGG embeddings → expected K-Means accuracy: 60–75%
- [ ] **HOG / SIFT Descriptors** — Texture-based features for better unsupervised separation
- [ ] **Full Dataset Training** — Train Random Forest on all 6,199 samples (currently 500-sample subset) → expected accuracy: 80–85%
- [ ] **Gaussian Mixture Models** — Probabilistic alternative to K-Means for non-spherical clusters
- [ ] **Data Augmentation** — Rotation, flipping, colour jitter to improve generalization

---

## 📚 References

1. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health.* arXiv:1511.08060
2. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32
3. MacQueen, J. B. (1967). *Some methods for classification of multivariate observations.* 5th Berkeley Symposium
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR, 12, 2825–2830
5. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with 🌿 for plant disease detection research

⭐ Star this repo if you found it useful!

</div>
