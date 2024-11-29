#logistic regression

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv(r"C:\Users\KUNAL\OneDrive\Desktop\important pdfs\WineQT.csv") 

# Inspect data
print("Columns in dataset:")
print(data.columns)

# Binning 'quality' into categories (low, medium, high)
bins = [0, 5, 7, 10]  # Adjust based on data distribution
labels = ['low', 'medium', 'high']
data['quality_category'] = pd.cut(data['quality'], bins=bins, labels=labels)

# Select features and target
X = data[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides"]]
Y = data["quality_category"]

# Encode categorical target
Y_encoded = Y.cat.codes  # Converts categories to integers

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.30, random_state=1)

# Train Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, Y_train)

# Predictions
Y_pred = log_reg_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy Score: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(Y_test, Y_pred, target_names=labels))

# Confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()