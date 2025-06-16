# CodeAlpha_Iris-Flower-Classification
Iris Species Classification Project
Overview
This project implements a machine learning model to classify Iris flower species (setosa, versicolor, virginica) based on their measurements (sepal length, sepal width, petal length, petal width). The model is built using Scikit-learn, evaluated on test data, and includes explanations of basic classification concepts. The dataset used is the classic Iris dataset, provided in Iris.csv.
The script (iris_classification_fixed.py) achieves 100% accuracy on the test set, demonstrating robust performance for this well-structured dataset.
Requirements

Python: Version 3.12 or higher
Libraries:
pandas
numpy
scikit-learn


Dataset: Iris.csv (included in the project directory)

Install the required libraries using pip:
pip install pandas numpy scikit-learn

Usage

Ensure the Dataset: Place Iris.csv in the project directory or update the file path in the script (iris_classification_fixed.py) to point to your CSV file.
Run the Script: Execute the script using Python 3.12:/usr/local/bin/python3.12 iris_classification_fixed.py


Output: The script will print:
NaN checks to confirm data cleanliness.
Model accuracy (1.00 or 100%).
Confusion matrix showing perfect classification (no misclassifications).
Classification report with precision, recall, and F1-score (all 1.00).
Basic classification concepts for educational purposes.



Script Details

Data Preprocessing:
Loads Iris.csv using pandas.
Checks for and handles missing values (NaN) using mean imputation for features and row removal for target.
Splits data into 80% training and 20% testing sets.
Standardizes features using StandardScaler.


Model: Uses a RandomForestClassifier with 100 estimators for robust classification.
Evaluation:
Accuracy: 100% on the test set (30 samples).
Confusion Matrix: Perfect classification with 10 setosa, 9 versicolor, and 11 virginica samples correctly predicted.
Classification Report: Precision, recall, and F1-score of 1.00 for all classes.


Educational Component: Includes explanations of key classification concepts (e.g., features, target, train-test split, evaluation metrics).

Results
The model achieved perfect performance on the test set:

Accuracy: 1.00 (100%)
Confusion Matrix:[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]


Classification Report:               precision    recall  f1-score   support
  Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      1.00      1.00         9
 Iris-virginica       1.00      1.00      1.00        11



This high accuracy is expected for the Iris dataset, which is well-separated, especially for setosa.
File Structure

Iris.csv: Dataset containing 150 rows of Iris measurements and species labels.
iris_classification_fixed.py: Main script for data preprocessing, model training, evaluation, and concept explanation.
README.md: This file, providing project documentation.

Notes

The Iris dataset is small and well-structured, leading to perfect accuracy. In real-world scenarios, lower accuracy might indicate overfitting or noisier data.
The script includes NaN handling for robustness, though the dataset had no missing values.
For further exploration, consider adding cross-validation or feature importance analysis (see script comments for suggestions).

License
This project is for educational purposes and uses the publicly available Iris dataset.
Contact
For questions or issues, please contact the project maintainer or open an issue in the repository.
