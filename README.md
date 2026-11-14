# 🌿 Plant Disease Classification

A deep learning project for identifying plant diseases from leaf images using **Custom CNN** and **MobileNetV2 Transfer Learning**.

The project covers dataset exploration, preprocessing, model training, evaluation, and eventually deployment through a Streamlit web application.

## 📌 Project Overview

Plant diseases can significantly affect agricultural productivity. This project aims to develop an image classification system that identifies plant diseases from leaf photographs.

The system focuses on five plant categories and 23 disease/health classes using the PlantVillage color image dataset.

### Supported Plants

- Apple
- Corn (Maize)
- Bell Pepper
- Potato
- Tomato

**Dataset statistics:**

| Description | Count |
|---|---:|
| Selected classes | 23 |
| Total images | 29,345 |
| Training images | 20,541 |
| Validation images | 4,402 |
| Testing images | 4,402 |

## 🧠 Models Developed

### 1. Custom CNN

A convolutional neural network developed from scratch using TensorFlow/Keras.

Architecture includes:
- Convolutional layers
- Batch normalization
- ReLU activation
- Max pooling
- Dropout regularization
- Global average pooling
- Dense classification layer

### 2. MobileNetV2

A transfer-learning model using ImageNet-pretrained MobileNetV2.

Approach:
- Pretrained MobileNetV2 feature extractor
- Frozen convolutional backbone
- Global average pooling
- Dense classification head
- Dropout regularization
- Class-weighted training

## 📊 Model Performance

Both models were evaluated using the same held-out test dataset of 4,402 images.

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Test Accuracy | 70.97% | **92.21%** |
| Test Loss | 1.0683 | **0.2583** |
| Macro Precision | 72.51% | **90.33%** |
| Macro Recall | 74.38% | **92.46%** |
| Macro F1-score | 69.67% | **91.21%** |
| Weighted F1-score | 72.96% | **92.21%** |

### Best Performing Model

MobileNetV2 achieved 92.21% test accuracy, outperforming the Custom CNN by 21.24 percentage points.

MobileNetV2 is currently selected as the best-performing model.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit (planned for deployment)
- Git and GitHub

## 📁 Project Structure

```text
Plant-Disease-Classification/
│
├── data/
│   └── class_distribution.csv
│
├── models/
│   └── (trained model files - stored separately)
│
├── outputs/
│   ├── evaluation/
│   ├── quality_check/
│   ├── splits/
│   ├── training/
│   └── visualizations/
│
├── src/
│   ├── check_images.py
│   ├── evaluate_custom_cnn.py
│   ├── evaluate_mobilenet.py
│   ├── explore_dataset.py
│   ├── preprocess_dataset.py
│   ├── split_dataset.py
│   ├── train_custom_cnn.py
│   ├── train_mobilenet.py
│   └── visualize_dataset.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/guptakumarsandip/Plant-Disease-Classification.git

cd Plant-Disease-Classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

The raw dataset is not included in this repository.

Obtain the PlantVillage color image dataset and place the selected class folders inside:

```text
data/raw/PlantVillage_color/
```

Keep the class folder names consistent with `data/class_distribution.csv`.

### 5. Run the project scripts

Explore and verify the dataset:

```bash
python src/visualize_dataset.py
python src/check_images.py
```

Split the dataset:

```bash
python src/split_dataset.py
```

Prepare the image pipelines:

```bash
python src/preprocess_dataset.py
```

Train the Custom CNN:

```bash
python src/train_custom_cnn.py
```

Train MobileNetV2:

```bash
python src/train_mobilenet.py
```

Evaluate the models:

```bash
python src/evaluate_custom_cnn.py
python src/evaluate_mobilenet.py
```

Run the scripts in the appropriate order. The split-generation step must be completed before training.

## 📈 Evaluation

The project uses:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Confusion matrix
- Per-class classification report

Evaluation results and training graphs are stored in the `outputs/` directory.

## ⚠️ Current Limitations

- The current experiments use PlantVillage images.
- Real-world photographs may contain different lighting, backgrounds, and image conditions.
- External dataset validation is still needed.
- The current MobileNetV2 model uses a frozen pretrained backbone.
- Streamlit deployment is planned but not yet implemented.

## 🚀 Future Improvements

- Fine-tune MobileNetV2.
- Build an interactive Streamlit application.
- Add image upload and disease prediction.
- Display prediction confidence and top-three predictions.
- Test on real-world leaf photographs.
- Improve model generalization and robustness.
- Deploy the application.

## 👨‍💻 Author

**Sandip Gupta**

GitHub: [@guptakumarsandip](https://github.com/guptakumarsandip)

## 📌 Project Status

**Current stage:** Model development and evaluation completed.

**Next milestone:** Streamlit web application development.

---

*This project is developed for educational and machine-learning experimentation purposes. Predictions should not be treated as a substitute for expert agricultural diagnosis.*