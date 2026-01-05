Here is the text you provided, reformatted into a clean, professional **Markdown** structure ready for GitHub.

I have organized the "Overview" section to be more scannable and turned the model performance stats into a clear comparison list.

### **Copy the Code Block Below**

```markdown
# 🛰️ Multimodal Property Valuation Using Satellite Imagery

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Architecture-Hybrid_CNN--MLP-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Project_Completed-success?style=for-the-badge)

## 🛠️ Libraries & Technologies Used

The project is built on a robust Python ecosystem for deep learning and image processing.

| Category | Library | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | `TensorFlow` / `Keras` | Core framework for building and training the Hybrid CNN-MLP model. |
| **Data Manipulation** | `Pandas` | Handling structured tabular data (CSVs) and data cleaning. |
| **Numerical Ops** | `NumPy` | Efficient matrix operations and array handling. |
| **Image Processing** | `Pillow (PIL)` | Loading, resizing (to 224x224), and processing satellite images. |
| **Visualization** | `Matplotlib` | Creating EDA plots, training loss curves, and Grad-CAM heatmaps. |
| **Preprocessing** | `Scikit-Learn` | `StandardScaler` for normalizing features and splitting datasets. |
| **API Handling** | `Requests` | Fetching satellite images via the Google Maps Static API. |

---

## 📂 Repository Structure

The project is modularized to ensure clear separation between data collection, processing, and modeling.

```bash
├── 📁 house_images/             # Directory containing fetched satellite images (224x224 RGB)
├── 📁 processed_data/           # Cleaned CSVs and feature-engineered datasets
│
├── 📜 data_fetcher.py           # Script: Interfaces with Google Maps Static API to download images
├── 📜 preprocessing.ipynb       # Notebook: Handles IQR outlier capping, scaling, and feature engineering
├── 📜 model_training.ipynb      # Notebook: Main training loop, Hybrid Architecture, and Grad-CAM
├── 📜 best_hybrid_model.keras   # Saved Model: Best performing weights from the training phase
├── 📜 final_submission_hybrid.csv # Output: Final price predictions for the test set
└── 📝 README.md                 # Documentation: Project overview and setup guide

```

---

## 📌 Project Overview & Key Topics

### 1. The Problem: "Context Blindness"

Traditional valuation models rely solely on spreadsheet numbers (bedrooms, square footage). They fail to "see" critical environmental factors like neighborhood density, greenery, road proximity, and privacy. This project solves this by giving the valuation model "eyes" via satellite imagery.

### 2. Solution: Hybrid Multimodal System

I developed a **Two-Stream Late Fusion Network** that mimics human appraisal intuition:

* **Visual Stream (CNN):** Uses `EfficientNetB0` (Frozen) to extract high-level visual features from satellite images.
* **Tabular Stream (MLP):** Uses a Dense Neural Network to process physical house specifications.
* **Fusion:** Combines both streams to predict the final property price.

### 3. Methodology & Data Pipeline

* **Data Acquisition:** Programmatically fetched **16,000+ satellite images** using property coordinates (lat, long).
* **Feature Engineering:** Created `house_age` and binary `is_renovated` features.
* **Preprocessing:** Applied **IQR Capping (Winsorization)** to remove extreme outliers in bedrooms/bathrooms and used `StandardScaler` for normalization.

### 4. Model Performance

Comparing the traditional approach vs. the hybrid approach:

| Model Type | Architecture | R² Score |
| --- | --- | --- |
| **Baseline** | Tabular Only (Random Forest) | 0.8479 |
| **Proposed** | **Tabular + Images (Multimodal)** | **0.8558** 🏆 |

### 5. Explainable AI (Grad-CAM)

To ensure the model is reliable, I used Grad-CAM visualizations to interpret what the CNN is "looking" at.

* **Result:** The heatmaps show "hotspots" (red/yellow) concentrated on building footprints and roofs.
* **Verification:** The model successfully learned to ignore irrelevant background noise like roads and shadows.

---

## ⚙️ Installation & Setup Guide

Follow these instructions to set up the environment and reproduce the results.

### 1. Prerequisites

* Python 3.8+
* Google Maps Static API Key (Required only for running `data_fetcher.py`).

### 2. Environment Setup

It is recommended to run this project in a virtual environment.

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

```

### 3. Install Dependencies

Run the following command to install all necessary libraries:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn pillow requests

```

---

## 🚀 How to Run the Project

Execute the files in the following order to replicate the pipeline.

### Step 1: Data Acquisition

* **File:** `data_fetcher.py`
* **Action:** Open the file, insert your Google Maps API key, and run the script.
```bash
python data_fetcher.py

```


* **Output:** Populates the `house_images/` directory with satellite images.

### Step 2: Preprocessing

* **File:** `preprocessing.ipynb`
* **Action:** Run all cells in the notebook.
* **Output:** Cleans the data, caps outliers, and saves `processed_train.csv` and `processed_test.csv` to `processed_data/`.

### Step 3: Model Training

* **File:** `model_training.ipynb`
* **Action:** Run all cells in the notebook.
* **Output:**
1. Trains the Hybrid CNN-MLP model.
2. Saves the best model to `best_hybrid_model.keras`.
3. Generates the `final_submission_hybrid.csv` file with price predictions.



```

---

### **Next Step**
Since you mentioned **Grad-CAM** in your text, do you have an actual screenshot of the heatmap (the image showing the "hotspots" on the roof)?

If you upload that image to your repository, I can give you the code snippet to display that image right inside your README under the "Explainable AI" section to make it look even more impressive. Would you like that?

```
