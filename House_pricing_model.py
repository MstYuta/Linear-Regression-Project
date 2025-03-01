import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("HousingData.csv")  # Ensure correct file path

# Handling missing data
df = df.assign(
    CRIM=df['CRIM'].fillna(df['CRIM'].mean()),
    ZN=df['ZN'].fillna(0),
    INDUS=df['INDUS'].fillna(df['INDUS'].mean()),
    CHAS=df['CHAS'].fillna(df['CHAS'].mode()[0]),
    AGE=df['AGE'].fillna(df['AGE'].median()),
    LSTAT=df['LSTAT'].fillna(df['LSTAT'].median())
)
df.drop(columns=['B'], inplace=True)  # Avoid potential bias

# Log transformation for skewed columns
columns_to_transform = ['CRIM', 'ZN', 'TAX', 'PTRATIO', 'LSTAT']
for col in columns_to_transform:
    df[col] = np.log1p(df[col])

# Define features and target
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial Features (for Linear Regression)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# PCA for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # Retains 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ---- MODEL 1: MULTIVARIATE LINEAR REGRESSION ----
class MultivariateLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=10000, lambda_=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_
        self.W = None
        self.b = None

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0
        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.W) + self.b
            error = y_pred - y
            dW = (1/m) * np.dot(X.T, error) + (self.lambda_ / m) * self.W
            db = (1/m) * np.sum(error)
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            if epoch % 1000 == 0:
                cost = (1/(2*m)) * np.sum(error**2) + (self.lambda_ / (2*m)) * np.sum(self.W**2)
                print(f"Epoch {epoch}: Cost = {cost:.4f}")

    def predict(self, X):
        return np.dot(X, self.W) + self.b

# Train and evaluate Linear Regression
lin_reg = MultivariateLinearRegression()
lin_reg.fit(X_train_poly, y_train)
y_pred_lin = lin_reg.predict(X_test_poly)
mse_lin = mean_squared_error(y_test, y_pred_lin)
print(f"Linear Regression Test MSE: {mse_lin:.4f}")

# ---- MODEL 2: RANDOM FOREST REGRESSOR ----
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_scaled, y_train)
y_pred_rf = rf_reg.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest Test MSE: {mse_rf:.4f}")

# ---- MODEL 3: XGBOOST REGRESSOR ----
xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_reg.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_reg.predict(X_test_scaled)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"XGBoost Test MSE: {mse_xgb:.4f}")

# ---- COMPARISON ----
print("\nModel Performance Comparison:")
print(f"Linear Regression MSE: {mse_lin:.4f}")
print(f"Random Forest MSE: {mse_rf:.4f}")
print(f"XGBoost MSE: {mse_xgb:.4f}")
