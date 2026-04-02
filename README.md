# Heart-Disease-Classifier
Heart Disease Classification Model, with several metrics of evaluation used.
# Heart Disease Prediction Model

This repository contains a Jupyter Notebook (`.ipynb`) that explores various machine learning models for predicting heart disease. The project focuses on data preprocessing, model selection, hyperparameter tuning, feature engineering, and performance evaluation to identify the most effective model for this classification task.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Explored](#models-explored)
- [Key Findings](#key-findings)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)

## Project Overview

The goal of this project is to build and evaluate machine learning models to predict the presence of heart disease based on a set of patient-specific features. The notebook walks through the entire machine learning pipeline, from initial data loading and exploration to advanced model tuning and interpretation.

## Dataset

The project uses the `heart-disease.csv` dataset, which contains various medical attributes and a target variable indicating the presence (1) or absence (0) of heart disease. The features include:
- `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure)
- `chol` (serum cholestoral in mg/dl), `fbs` (fasting blood sugar > 120 mg/dl)
- `restecg` (resting electrocardiographic results), `thalach` (maximum heart rate achieved)
- `exang` (exercise induced angina), `oldpeak` (ST depression induced by exercise relative to rest)
- `slope` (the slope of the peak exercise ST segment), `ca` (number of major vessels (0-3) colored by flourosopy)
- `thal` (thalassemia: 1 = normal; 2 = fixed defect; 3 = reversible defect)

## Methodology

The notebook follows a structured machine learning approach:

1.  **Data Loading and Initial Exploration**: The dataset is loaded using pandas, and basic exploratory data analysis (EDA) is performed.
2.  **Data Splitting**: The data is split into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) with `test_size=0.2` and `np.random.seed` for reproducibility.
3.  **Baseline Model Evaluation**: Initial performance of Logistic Regression, K-Nearest Neighbors (KNN), and RandomForest Classifier is assessed without extensive tuning.
4.  **Hyperparameter Tuning**: 
    -   **KNN**: `GridSearchCV` is used to find the optimal `n_neighbors` for the `KNeighborsClassifier`.
    -   **Logistic Regression**: `GridSearchCV` is applied to tune the `C` parameter and `solver` for `LogisticRegression`.
5.  **Feature Scaling**: `StandardScaler` is applied to numerical features, especially for distance-based models like KNN and Logistic Regression, to improve performance.
6.  **Feature Importance and Selection**: Feature importances for Logistic Regression are derived from the absolute values of its coefficients. Less important features are identified and removed to evaluate the impact on model performance.
7.  **Model Re-evaluation**: Models are retrained and re-evaluated after feature scaling and feature reduction steps.
8.  **Expanded Hyperparameter Tuning**: An expanded `GridSearchCV` is implemented for Logistic Regression, exploring different `penalty` types (`l1`, `l2`), `solvers` (`liblinear`, `lbfgs`, `saga`), and `C` values to find the global best configuration on scaled data.
9.  **Prediction on New Data**: The best-performing model (`gs_log_reg_reduced`) is demonstrated by making predictions on synthetic, unseen data, emphasizing the need for consistent preprocessing (feature reduction).
10. **Model Evaluation Metrics**: Performance is evaluated using accuracy, confusion matrices, and ROC curves.

## Models Explored

-   **Logistic Regression**: A linear model used for binary classification. Tuned using `GridSearchCV`.
-   **K-Nearest Neighbors (KNN)**: A non-parametric, distance-based classifier. Tuned using `GridSearchCV`.
-   **RandomForest Classifier**: An ensemble learning method for classification. Initial performance assessed.

## Key Findings

-   Initial evaluation provided baseline accuracy scores for all models.
-   Hyperparameter tuning significantly improved KNN performance.
-   Feature scaling using `StandardScaler` was applied to optimize distance-based models.
-   Feature importance analysis with Logistic Regression coefficients revealed that removing the 4 least important features (`thalach`, `trestbps`, `age`, `chol`) slightly improved the test set score for Logistic Regression, demonstrating the potential benefits of feature selection.
-   Expanded hyperparameter tuning for Logistic Regression on scaled features further refined the model, achieving the best cross-validation score, though the test score for the expanded model was slightly lower than the reduced features model in this specific run.
-   The Logistic Regression model, particularly after feature reduction, showed strong predictive capabilities and was used for making predictions on new, unseen data.

## Dependencies

To run this notebook, you will need the following Python libraries:

-   `pandas`
-   `numpy`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn` (`sklearn`)


How to Run
Clone the repository:

git clone <your-repository-url>
cd <your-repository-name>
Ensure dependencies are installed (see [redacted link]).

Open the Jupyter Notebook: You can open the .ipynb file using Jupyter Lab, Jupyter Notebook, or Google Colab.

Google Colab: Upload the notebook to Google Colab, or open it directly if stored on GitHub.
Local Environment: Run jupyter notebook or jupyter lab in your terminal and navigate to the notebook file.
Execute Cells: Run the cells sequentially to reproduce the analysis and model training.
