# Tuberculosis Detection using Deep Learning (Streamlit App)

This project is an interactive web application built using **Streamlit** that detects **Tuberculosis (TB)** from **chest X-ray images** using a trained deep learning model. It supports prediction, confidence scoring, and Grad-CAM visualization for explainability.

---

## Features

* Upload a chest X-ray (JPG/PNG)
* Get a binary classification: **TB Positive / TB Negative**
* View the **prediction confidence**
* See **Grad-CAM heatmap** for model explainability

---

## Folder Structure

```
tb_detection_app/
├── app.py                       # Main Streamlit application
├── model/
│   └── tb_model.h5             # Pre-trained Keras model file
├── utils/
│   ├── preprocess.py           # Preprocessing pipeline
│   └── gradcam.py              # Grad-CAM generation
├── sample_images/              # Optional sample test images
└── requirements.txt            # Python dependencies
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourname/tb_detection_app.git
cd tb_detection_app
```

### 2. Set up virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place your trained model

Make sure your trained model `tb_model.h5` is inside the `model/` directory. If you're missing it, export from your training notebook:

```python
model.save("tb_model.h5")
```

---

## Running the App

```bash
streamlit run app.py
```

This will open a web browser at `http://localhost:8501`.

---

## Sample Usage

1. Upload a chest X-ray image.
2. The app will display:

   * Model's prediction (TB Positive or TB Negative)
   * Confidence score
   * Grad-CAM overlay for model interpretability

---

## Model Details

* Architecture: DenseNet121 (fine-tuned on TB dataset)
* Input size: 224x224
* Dataset: [Kaggle TB Chest X-ray Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
* Evaluation: Accuracy, F1-score, AUC, Confusion Matrix

---

## Future Improvements

* Batch prediction support
* REST API using FastAPI
* Deployment on Hugging Face Spaces or Render
* Integration with real hospital systems via HL7/FHIR

---

## License

MIT License

---

## Acknowledgements

* TensorFlow & Keras for model building
* Streamlit for rapid UI development
* Grad-CAM for model explainability
* Kaggle for providing the open-access dataset
