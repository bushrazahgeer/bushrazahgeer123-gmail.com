---
# 🥭 **MANGO LEAF DISEASE DETECTION USING VGG16 CNN**

Automated detection of mango leaf diseases using **VGG16 CNN with Transfer Learning**.
Achieves **97%+ accuracy** across **8 disease categories**, enabling **early detection** for better crop yield.

---

## ⚙ System Requirements

> Training can be done on mid-range hardware or free cloud GPUs (Google Colab / Kaggle Notebooks).

**Minimum (CPU-only):**

* RAM: 8 GB
* CPU: Quad-core i5 / Ryzen 5

**Recommended (GPU training):**

* RAM: 16 GB
* GPU: NVIDIA GTX 1650 / RTX 3050+ (CUDA-enabled)
* Python: 3.10
* Environment: Anaconda (virtual env recommended)

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

**Source:** [Kaggle – Mango Leaf Disease Dataset](https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset)
**Size:** ~4,000 images
**Classes (8):** Anthracnose · Bacterial Canker · Cutting Weevil · Die Back · Gall Midge · Healthy · Powdery Mildew · Sooty Mould

**Sample Images:**
![Sample Leaves](https://github.com/bushrazahgeer/bushrazahgeer123-gmail.com/blob/4024821471858432b5b8fdf1667fded9cbb43c19/sample_image.png)
![Dataset Overview](https://github.com/bushrazahgeer/bushrazahgeer123-gmail.com/blob/4024821471858432b5b8fdf1667fded9cbb43c19/SAMPLE_imgs.png)

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

### 2️⃣ Model Architecture: VGG16 + Custom Head

* **Base Model:** VGG16 (pre-trained on ImageNet)
* **Custom Layers:**

  * GlobalAveragePooling2D
  * Dense (1024 neurons, ReLU, L2 regularization)
  * Dropout (0.6)
  * Dense (Softmax, 8 classes)
* **Optimizer:** Adam (lr=0.0001)
* **Loss:** Categorical Crossentropy
* **Batch Size:** 16
* **Epochs:** 10

**Architecture Diagram (Sample):**
![Confusion Matrix](https://github.com/bushrazahgeer/bushrazahgeer123-gmail.com/blob/4024821471858432b5b8fdf1667fded9cbb43c19/c_matrix.png)

---

## 📈 Training & Results

### 🔹 Training Performance

| Metric   | Training | Validation |
| -------- | -------- | ---------- |
| Accuracy | 100%     | 99.87%     |
| Loss     | → 0      | ~0.01      |

**Training Curves:**
![Accuracy Curve](https://github.com/bushrazahgeer/bushrazahgeer123-gmail.com/blob/4024821471858432b5b8fdf1667fded9cbb43c19/accuracy.png)

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

**Sample Predictions:**
![Prediction Examples](https://github.com/bushrazahgeer/bushrazahgeer123-gmail.com/blob/4024821471858432b5b8fdf1667fded9cbb43c19/o_identify.png)
![Prediction Examples](https://github.com/bushrazahgeer/bushrazahgeer123-gmail.com/blob/4024821471858432b5b8fdf1667fded9cbb43c19/op_identify.png)

---

## ⚙️ Installation & Usage

```bash
# Create virtual environment
conda create -n mango_env python=3.10
conda activate mango_env

# Install dependencies
pip install -r requirements.txt

# Run inference
python predict.py --image path/to/mango_leaf.jpg
```

---

## ✨ Features

| Feature                       | Description                                          |
| ----------------------------- | ---------------------------------------------------- |
| 🍃 Multi-class Classification | Detects **8 mango leaf categories**                  |
| ⚡ High Accuracy               | Achieves **97% test accuracy**                       |
| 📊 Visualization              | Training curves, confusion matrix, prediction scores |
| 🔄 Extensible                 | Easily adaptable for other plants/diseases           |

---

## 🏁 Conclusion

Deep learning can transform agriculture 🌾. This **VGG16-based model** enables **early detection** of mango leaf diseases, reducing losses and boosting yield.

**Next Steps:** Extend to other crops for **AI-driven precision agriculture** 🚀.

---

## 🧑‍💻 Author

**Bushra Zahgeer**
📧 [bushrazahgeer123@gmail.com](mailto:bushrazahgeer123@gmail.com)

⭐ If you find this project useful, give it a star to support! ⭐

---
