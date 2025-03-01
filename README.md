# Linear Regression Project

## 📌 Overview

This project implements **Multivariate Linear Regression** from scratch using NumPy and compares its performance with **Random Forest** and **XGBoost** models. The dataset used is **Boston Housing Data**, and various preprocessing techniques, including **log transformation, feature scaling, and PCA**, are applied.

## 🚀 Features

- Implements **Linear Regression from scratch** with L2 regularization.
- Compares performance with **Random Forest Regressor** and **XGBoost Regressor**.
- **Polynomial Feature Engineering** to improve model performance.
- **Principal Component Analysis (PCA)** to retain 95% variance.
- **Feature scaling using StandardScaler** to normalize data.

## 📂 Dataset

The project uses the **Boston Housing Dataset**, which contains **506 instances** with **14 features** related to housing prices in Boston.

## 🛠️ Installation

Before running the project, install the required libraries:

```bash
pip install numpy pandas scikit-learn xgboost
```

## 🏗️ Project Structure

```
Linear-Regression-Project/
│── HousingData.csv           # Dataset file
│── House_pricing_model.py    # Main Python script
│── README.md                 # Project documentation
```

## 📊 Model Implementation

### 1️⃣ **Data Preprocessing**

- Handle missing values using **mean, median, and mode**.
- Apply **log transformation** on highly skewed features.
- Scale features using **StandardScaler**.
- Generate **Polynomial Features (Degree 2)**.
- Apply **PCA** to reduce dimensions.

### 2️⃣ **Multivariate Linear Regression from Scratch**

- Implemented using **Gradient Descent with L2 Regularization**.
- Cost function: **Mean Squared Error (MSE) with L2 penalty**.

### 3️⃣ **Random Forest Regressor**

- Implemented using **sklearn.ensemble.RandomForestRegressor**.
- Default hyperparameters used.

### 4️⃣ **XGBoost Regressor**

- Implemented using **xgboost.XGBRegressor**.
- Default hyperparameters used.

## 📈 Results Comparison

| Model                             | Test MSE |
| --------------------------------- | -------- |
| **Custom Linear Regression**      | 15.7651  |
| **Random Forest Regressor**       | 8.2915   |
| **XGBoost Regressor**             | 6.6369   |

## 🏃‍♂️ Running the Project

To run the project, execute:

```bash
python House_pricing_model.py
```

