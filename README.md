# ✋ Hand Sign Recognition (A–Z)

A real-time **Hand Sign Recognition System** built using **OpenCV, CVZone, and Machine Learning**.

The system detects hand gestures from a webcam and predicts the corresponding **alphabet (A–Z)**.

---

## 🚀 Features

* Real-time webcam hand detection
* Dataset creation for sign language
* Machine learning model training
* Alphabet prediction (A–Z)
* ~97% model accuracy

---

## 🧠 Tech Stack

* Python
* OpenCV
* CVZone
* NumPy
* Scikit-Learn
* TensorFlow (experimental)

---

## 📁 Project Structure

```
hand-sign-recognition
│
├── data/                  # Dataset (A–Z folders)
│   ├── A/
│   ├── B/
│   ├── C/
│
├── scripts/
│   ├── collect_data.py    # Capture hand images
│   ├── train.ipynb        # Train ML model
│   └── predict.py         # Real-time prediction
│
├── model/
│   └── hand_sign_model.pkl
│
├── test/
│   └── A/
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The dataset contains **hand gesture images for all English alphabets (A–Z)**.

Each folder corresponds to a single alphabet.

Example:

```
data/
 ├── A/
 ├── B/
 ├── C/
```

---

## 🧪 Model Training

Images are processed using:

1. Grayscale conversion
2. Resize to **64x64**
3. Flatten to feature vector

A **Support Vector Machine (SVM)** classifier is trained.

Model Accuracy:

```
97%+
```

---

## ▶️ Run the Project

Install dependencies:

```
pip install -r requirements.txt
```

Collect dataset:

```
python scripts/collect_data.py
```

Train model:

```
python scripts/train.ipynb
```

Run prediction:

```
python scripts/predict.py
```

---

## 🔮 Future Improvements

* CNN-based model
* Word-level prediction
* Sign language translator
* Web deployment

---

## 👨‍💻 Author

**Anurag Sharma**
B.Tech CSE | Machine Learning Developer

