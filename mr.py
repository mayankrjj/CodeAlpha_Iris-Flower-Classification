 #Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Loading the Iris dataset
# Replace 'Iris.csv' with the actual path to your CSV file
df = pd.read_csv('Iris.csv')

# Checking for NaN values
print("NaN in dataset before cleaning:\n", df.isna().sum())

# Preparing features (X) and target (y)
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Handling NaN values in features
imputer = SimpleImputer(strategy='mean')  # Impute with mean for numerical columns
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Handling NaN values in target (remove rows with NaN in Species)
if y.isna().sum() > 0:
    print(f"Removing {y.isna().sum()} rows with NaN in target variable.")
    non_na_indices = y.notna()
    X = X[non_na_indices]
    y = y[non_na_indices]

# Verify no NaN values remain
print("\nNaN in features after imputation:\n", X.isna().sum())
print("NaN in target after cleaning:\n", y.isna().sum())

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Printing the results
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Basic explanation of classification concepts
print("\nBasic Classification Concepts:")
print("1. Classification: Predicting a categorical label (here, Iris species).")
print("2. Features: Measurable properties (sepal/petal length/width).")
print("3. Target: The variable to predict (Species: setosa, versicolor, virginica).")
print("4. Train-Test Split: Dividing data to train the model and test its performance.")
print("5. Standardization: Scaling features to have mean=0 and variance=1 for better model performance.")
print("6. Random Forest: An ensemble method using multiple decision trees for robust predictions.")
print("7. Evaluation Metrics:")
print("   - Accuracy: Proportion of correct predictions.")
print("   - Confusion Matrix: Table showing true vs. predicted labels.")
print("   - Precision: Proportion of positive predictions that were correct.")
print("   - Recall: Proportion of actual positives correctly identified.")
print("   - F1-Score: Harmonic mean of precision and recall.")
print("8. Imputation: Handling missing values by replacing them (e.g., with mean) or removing rows.")
