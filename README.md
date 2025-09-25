---

# 🥭 Mango Leaf Disease Detection using VGG16 CNN


---
---

⚙ System Requirements

This project is moderately resource-intensive. You don’t need a workstation-class system—training can be done on mid-range hardware or cloud GPUs.

Minimum Specs (for local CPU training):

RAM: 8 GB

CPU: Quad-core i5 / Ryzen 5


Recommended Specs (for faster GPU training):

RAM: 16 GB

GPU: NVIDIA GTX 1650 / RTX 3050 or higher (CUDA-enabled)

Python: 3.10

Environment: Anaconda (virtual environment recommended)


💡 If you don’t have a GPU, use Google Colab or Kaggle Notebooks for free GPU access.


---
## 🛠️ Tech Stack

**Backend & ML:**

* Python 3.10
* TensorFlow 2.12 / Keras
* OpenCV & Pillow (Image Processing)
* NumPy & Pandas (Data Handling)
* Scikit-learn (Evaluation Metrics)

**Visualization:**

* Matplotlib & Seaborn (Plots, Confusion Matrix)

**Environment:**

* Anaconda / Miniconda (for virtual environment and dependency management)

---

## 📖 Methodology

### 1. Dataset

The model is trained on the **Mango Leaf Disease Dataset** available on Kaggle:

* **Link:** [Kaggle - Mango Leaf Disease Dataset](https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset)
* **Classes:**

  * Anthracnose
  * Bacterial Canker
  * Cutting Weevil
  * Die Back
  * Gall Midge
  * Healthy
  * Powdery Mildew
  * Sooty Mould
* **Number of Images:** ~4,000 (Train + Test)

---
https://github.com/bushrazahgeer/bushrazahgeer123-gmail.com/blob/4024821471858432b5b8fdf1667fded9cbb43c19/sample_image.png
https://github.com/bushrazahgeer/bushrazahgeer123-gmail.com/blob/4024821471858432b5b8fdf1667fded9cbb43c19/SAMPLE_imgs.png
### 2. Data Preprocessing

* Images resized to **256x256 pixels**.
* Pixel values normalized to **range 0–1**.
* Labels converted to **one-hot encoding** for categorical classification.
* **Train/Test split:** 95% training, 5% testing.
* Sample visualization of training and testing images is included below.


---

### 3. Model Architecture: VGG16 Transfer Learning

* **Base Model:** Pre-trained VGG16 on ImageNet.
* **Modifications:**

  * Removed final classification layers.
  * Added **GlobalAveragePooling2D** layer.
  * Added **Dense layer (1024 neurons)** with ReLU and L2 regularization.
  * **Dropout layers** (rate=0.6) to reduce overfitting.
  * Final **Dense layer** with softmax activation for 8 classes.

* **Optimizer:** Adam (learning rate = 0.0001)
* **Loss Function:** Categorical Crossentropy
* **Batch Size:** 16
* **Epochs:** 10
https://github.com/bushrazahgeer/bushrazahgeer123-gmail.com/blob/4024821471858432b5b8fdf1667fded9cbb43c19/c_matrix.png
---

### 4. Training Results

The model was trained using **80% of the dataset for training** and **20% for validation**. The training progress is shown below.

https://github.com/bushrazahgeer/bushrazahgeer123-gmail.com/blob/4024821471858432b5b8fdf1667fded9cbb43c19/accuracy.png

* **Training Accuracy:** 100%
* **Validation Accuracy:** 99.87%
* The model converges quickly due to transfer learning and pre-trained features.

---
## 📌 Overview

Mango farming is vital 🌳, but leaf diseases like **Anthracnose, Powdery Mildew, Bacterial Canker,** and **Sooty Mould** often harm yield & quality.
Traditional **manual inspection** is:
❌ Time-consuming
❌ Error-prone
❌ Reactive (late detection)
https://github.com/bushrazahgeer/bushrazahgeer123-gmail.com/blob/4024821471858432b5b8fdf1667fded9cbb43c19/o_identify.png
https://github.com/bushrazahgeer/bushrazahgeer123-gmail.com/blob/4024821471858432b5b8fdf1667fded9cbb43c19/op_identify.png
✅ **Our Solution:** A **VGG16 CNN with Transfer Learning** that detects **8 classes** of mango leaves with **97%+ accuracy**.

---

## 🛠️ Tech Stack

| Category             | Tools / Frameworks      |
| -------------------- | ----------------------- |
| **Language**         | Python 3.10             |
| **Deep Learning**    | TensorFlow 2.12 · Keras |
| **Image Processing** | OpenCV · Pillow         |
| **Data Handling**    | NumPy · Pandas          |
| **Visualization**    | Matplotlib · Seaborn    |
| **Environment**      | Anaconda / Miniconda    |

---

## 📂 Dataset

| Property        | Details                                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Source**      | [Kaggle – Mango Leaf Disease Dataset](https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset)       |
| **Size**        | ~4,000 images                                                                                                    |
| **Classes (8)** | Anthracnose · Bacterial Canker · Cutting Weevil · Die Back · Gall Midge · Healthy · Powdery Mildew · Sooty Mould |


---

## 🔬 Methodology

### 1️⃣ Data Preprocessing

| Step      | Description                          |
| --------- | ------------------------------------ |
| Resize    | All images resized to **256×256 px** |
| Normalize | Pixel values scaled **0–1**          |
| Encoding  | Labels → One-hot encoding            |
| Split     | Train/Test = **95% / 5%**            |

---

### 2️⃣ Model Architecture (VGG16 + Custom Head)

| Layer         | Details                                                                                      |
| ------------- | -------------------------------------------------------------------------------------------- |
| Base Model    | VGG16 (pre-trained on ImageNet)                                                              |
| Custom Layers | GlobalAveragePooling2D · Dense (1024, ReLU, L2) · Dropout (0.6) · Dense (Softmax, 8 classes) |
| Optimizer     | Adam (lr=0.0001)                                                                             |
| Loss          | Categorical Crossentropy                                                                     |
| Batch Size    | 16                                                                                           |
| Epochs        | 10                                                                                           |



---

## 📈 Results

### 🔹 Training Performance


| Metric   | Training | Validation |
| -------- | -------- | ---------- |
| Accuracy | 100%     | 99.87%     |
| Loss     | → 0      | ~0.01      |

---

### 🔹 Test Evaluation

📊 **Overall Test Accuracy:** ~97%
| Class            | Precision | Recall | F1-Score |
| ---------------- | --------- | ------ | -------- |
| Anthracnose      | 0.97      | 1.00   | 0.98     |
| Bacterial Canker | 0.95      | 1.00   | 0.98     |
| Cutting Weevil   | 1.00      | 0.97   | 0.98     |
| Die Back         | 1.00      | 1.00   | 1.00     |
| Gall Midge       | 1.00      | 0.93   | 0.96     |
| Healthy          | 0.93      | 1.00   | 0.97     |
| Powdery Mildew   | 1.00      | 0.90   | 0.95     |
| Sooty Mould      | 0.90      | 1.00   | 0.95     |

---

### 🔹 Sample Predictions
---

## ⚙️ Installation & Usage

### 🔹 Requirements

| Resource | Recommended                                 |
| -------- | ------------------------------------------- |
| OS       | Windows / macOS / Linux                     |
| RAM      | 16 GB                                       |
| GPU      | NVIDIA GTX 1660+ (RTX 3060+ 🚀 recommended) |
| Python   | 3.10                                        |

### 🔹 Setup

```bash
# Create environment
conda create -n mango_env python=3.10
conda activate mango_env

# Install dependencies
pip install -r requirements.txt
```

### 🔹 Run Inference

```bash
python predict.py --image path/to/mango_leaf.jpg
```

---

## ✨ Features

| Feature                       | Description                                            |
| ----------------------------- | ------------------------------------------------------ |
| 🍃 Multi-class Classification | Detects **8 mango leaf categories**                    |
| ⚡ High Accuracy               | Achieves **97% test accuracy**                         |
| 📊 Visualization              | Training curves · Confusion matrix · Prediction scores |
| 🔄 Extensible                 | Easily adaptable for other plants/diseases             |

---

## 🏁 Conclusion

This project shows how **deep learning (VGG16)** can transform agriculture 🌾.
With **97% accuracy**, our model enables **early detection**, reducing losses & boosting yield.

👉 The workflow can be scaled to other crops → **AI-driven precision agriculture** is here! 🚀

---

## 🧑‍💻 Author

**👩‍💻 Bushra Zahgeer**
📧 [bushrazahgeer123@gmail.com](mailto:bushrazahgeer123@gmail.com)

---

  ⭐ If you find this project useful, give it a star to support! ⭐

---
