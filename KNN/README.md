# K-Nearest Neighbors (KNN) Model Documentation 🧮

## Overview 🔍
This notebook implements a K-Nearest Neighbors classifier for cancer classification, optimizing the model through hyperparameter tuning and addressing class imbalance with SMOTE.

## Features ⭐
- Implementation of KNN algorithm with various distance metrics
- Hyperparameter optimization
- K-value selection through F1 score and error rate analysis
- Class imbalance handling with SMOTE
- Comprehensive model evaluation

## Step-by-Step Workflow 📋

### 1. Setup and Data Preparation 📥
- Imports necessary libraries (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Sets random seed for reproducibility
- Loads the pre-cleaned cancer dataset
- Separates features (X) and target variable (y)
- Splits data into training and testing sets with stratification

### 2. K Parameter Optimization 🔢
- Tests K values from 1 to 50
- Calculates F1 scores for each K value
- Plots F1 scores to identify optimal K
- Creates an elbow curve plotting error rates vs K values
- Determines that K=2 provides the best performance

### 3. Model Building 🏗️
- Implements KNeighborsClassifier with the optimal K value
- Trains the model on the training data
- Creates a custom evaluation function to calculate metrics

### 4. Hyperparameter Tuning ⚙️
- Performs grid search across multiple parameters:
  - Weights (uniform, distance)
  - Distance metrics (euclidean, manhattan, minkowski)
  - P parameter values (1, 2)
- Uses cross-validation to find optimal parameters
- Evaluates the best model on both training and test data

### 5. Model Evaluation 📊
- Generates classification reports
- Displays confusion matrix
- Calculates accuracy, precision, recall, and F-score

### 6. Class Imbalance Handling ⚖️
- Applies SMOTE to address class imbalance in the training data
- Visualizes the balanced class distribution
- Retrains the model on the balanced data
- Evaluates the final model performance

## Model Parameters ⚙️
- **n_neighbors**: 2 (optimal value determined through analysis)
- **weights**: Distance-based weighting or uniform (determined by grid search)
- **metric**: Euclidean, Manhattan, or Minkowski (determined by grid search)

## Performance Metrics 📏
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F-score (with beta=5 to emphasize recall for cancer detection)
- Confusion Matrix

## Results 🏆
The notebook provides:
- Comparison of model performance before and after SMOTE
- Visualizations of model metrics
- Training vs. test set performance analysis

## Usage Instructions 📝
1. Ensure the cleaned dataset is available in the specified path
2. Run all cells sequentially
3. Examine the plots to understand K-value selection
4. Review the grid search results to understand hyperparameter selection
5. Analyze the final model performance metrics

## Key Insights 💡
- KNN performs well for cancer classification with properly tuned parameters
- The model achieves improved performance with class balancing
- Distance-based metrics significantly impact model performance
