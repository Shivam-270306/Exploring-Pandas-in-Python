#KNNN
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv(r"C:\Users\KUNAL\OneDrive\Desktop\important pdfs\WineQT.csv") 

# Inspect data
print("Columns in dataset:")
print(data.columns)
print("\nCorrelation matrix:")
print(data.corr().to_string())

# Select features and target
X = data[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides"]]
Y = data["quality"]



# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

# Train KNN model
k = 5  # Number of neighbors
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X_train, Y_train)

# Predictions
Y_pred = knn_model.predict(X_test)

# Metrics
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualizing Predictions vs Actual
plt.scatter(Y_test, Y_pred, alpha=0.6, color='blue', label="Predicted vs Actual")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color="red", label="Ideal Prediction Line")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("KNN Regression: Actual vs Predicted")
plt.legend()
plt.show()