 Apple Leaf Disease Classification using Random Forest
A machine learning project for classifying apple leaf diseases using traditional computer vision techniques and a Random Forest classifier.
_____________________________________________________________________________________________________________________________________________________

 🌱 Project Overview

Plant diseases significantly affect agricultural productivity. Early detection of leaf diseases helps farmers take preventive actions and reduce crop loss.
This project builds a machine learning pipeline that classifies apple leaf images into disease categories using image preprocessing and an ensemble learning algorithm.
The system extracts pixel-level features from images and predicts the disease class using a Random Forest model.
_____________________________________________________________________________________________________________________________________________________

🍃 Disease Categories

The model classifies apple leaves into the following categories:

* Apple Scab
* Apple Black Rot
* Apple Cedar Apple Rust
* Healthy Apple Leaf
_____________________________________________________________________________________________________________________________________________________

📂 Dataset

The dataset consists of labeled apple leaf images organized into class-specific folders. Each folder corresponds to a disease category.
Example dataset structure:

```
dataset/
    Apple___Apple_scab/
    Apple___Black_rot/
    Apple___Cedar_apple_rust/
    Apple___healthy/
```
Each image represents a sample and the folder name acts as the class label.

_____________________________________________________________________________________________________________________________________________________

⚙️ Machine Learning Pipeline

The project follows a structured workflow:

```
Image Dataset
      ↓
Image Preprocessing
      ↓
Feature Extraction
      ↓
Label Encoding
      ↓
Train-Test Split
      ↓
Random Forest Training
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
Prediction on New Images
```

_____________________________________________________________________________________________________________________________________________________

 🖼 Image Preprocessing

To standardize the dataset and reduce computational cost:

* Images resized to **64 × 64 pixels**
* Converted from **BGR to RGB format**
* Flattened into **12,288-dimensional feature vectors**

This converts images into numerical data suitable for machine learning algorithms.
_____________________________________________________________________________________________________________________________________________________

🤖 Model

The project uses a **Random Forest Classifier** from Scikit-Learn.
Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting.

Advantages include:
* Robust performance
* Reduced overfitting
* Effective for high-dimensional data

_____________________________________________________________________________________________________________________________________________________

🔧 Hyperparameter Tuning

Model parameters were optimized using **GridSearchCV with 5-fold cross-validation**.
Parameters explored:

```
n_estimators: [50, 100]
max_depth: [None, 10]
```

Best parameters obtained:

```
n_estimators = 100
max_depth = 10
```
_____________________________________________________________________________________________________________________________________________________

📊 Model Performance

Test Accuracy:

```
75%
```

Classification metrics:

| Class                  | Precision | Recall | F1 Score |
| ---------------------- | --------- | ------ | -------- |
| Apple Scab             | 0.69      | 0.77   | 0.73     |
| Apple Black Rot        | 0.76      | 0.77   | 0.77     |
| Apple Cedar Apple Rust | 0.83      | 0.73   | 0.78     |
| Apple Healthy          | 0.80      | 0.70   | 0.74     |

Weighted Average F1 Score: **0.75**
_____________________________________________________________________________________________________________________________________________________
 🗂 Repository Structure

```
apple-leaf-disease-classification
│
├── docs/
│   ├── dataset.md
│   ├── results.md
│   └── workflow.png
│
├── notebooks/
│   └── Project.ipynb
│
├── reports/
│   └── Apple Disease Report.pdf
│
├── src/
│   ├── train_model.py
│   ├── preprocess.py
│   ├── predict.py
│   └── evaluate_model.py
│
├── README.md
├── requirements.txt
└── .gitignore
```
_____________________________________________________________________________________________________________________________________________________

🚀 Installation

Clone the repository:

```
git clone https://github.com/yourusername/apple-leaf-disease-classification.git
```

Navigate to the project folder:

```
cd apple-leaf-disease-classification
```

Install required dependencies:

```
pip install -r requirements.txt
```
_____________________________________________________________________________________________________________________________________________________

 ▶️ Running the Project

Open the notebook environment:

```
jupyter notebook
```

Run the notebook to train the model and evaluate results.

_____________________________________________________________________________________________________________________________________________________

🔍 Example Prediction

The trained model can classify new apple leaf images.
Example output:

```
Input Image: leaf.jpg
Prediction: Apple Cedar Apple Rust
```
_____________________________________________________________________________________________________________________________________________________

📈 Future Improvements

Possible improvements include:

* Training Convolutional Neural Networks (CNNs)
* Applying data augmentation
* Expanding the dataset
* Deploying the model as a web application
* Developing a mobile disease detection tool for farmers
________________________________________________________________________________________________________________________________________________________________________________________________________

