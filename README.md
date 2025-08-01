# Loan-Default-Prediction
The primary objective of this project is to develop and evaluate robust machine learning models capable of predicting the likelihood of loan default by borrowers. By accurately identifying high-risk applicants, the project aims to empower financial institutions to make more informed lending decisions and mitigate potential financial losses.
Loan Default Prediction Project
# Overview
This project focuses on building and evaluating machine learning models to predict loan defaults. By leveraging historical financial data, the goal is to develop a robust predictive tool that can help financial institutions assess risk more accurately, minimize losses from non-performing loans, and optimize their lending strategies.

# Project Objective
The primary objective is to predict whether a loan applicant will default on their loan (Status = 1) or not (Status = 0). This is a critical classification problem for risk management in the financial sector.

# Problem Statement: The Cost of Defaults
Loan defaults pose a significant financial burden on lenders. Traditional risk assessment methods may not capture complex patterns in applicant data, leading to missed high-risk individuals. This project aims to mitigate these risks by employing advanced data analysis and machine learning techniques to forecast default likelihood. I used a probability threshold of 0.25 to reduce false negatives which are extremely costly for banks.

# Methodology & Key Steps
The project follows a comprehensive data science pipeline, emphasizing both data analysis and machine learning:

# Data Acquisition & Initial Exploration:

Loaded raw loan application data.

Performed initial exploratory data analysis (EDA) to understand data distributions, identify outliers, and detect missing values. Performed Univariate Analysis then Bivariate Analysis

Dataset Download: The dataset used for this project can be downloaded from Kaggle: [https://www.kaggle.com/datasets/yashpaloswal/loan-default-prediction](https://www.kaggle.com/datasets/yasserh/loan-default-dataset?resource=download)

# Data Preprocessing & Feature Engineering:

Missing Value Imputation: Handled missing values in numerical columns using median and mode imputation.

Outlier Treatment & Transformation: Identified and transformed highly skewed numerical features (income, LTV, property_value, Upfront_charges, loan_amount) using capping and flooring then log1p or sqrt transformations to achieve more symmetrical distributions. Original untransformed columns were then removed.

Categorical Feature Encoding: Converted all categorical variables into numerical dummy variables using one-hot encoding (pd.get_dummies). Boolean True/False values from encoding were converted to 1/0.

Multicollinearity Handling: Analyzed the correlation matrix to identify and remove highly correlated/redundant features (e.g., pairs of perfectly inverse or identical dummy variables) to improve model stability.

Handling Class Imbalance:

Identified a significant class imbalance in the target variable (Status), with the "No Default" class being the majority (~81%) and "Default" being the minority (~19%).

Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training dataset to balance the class distribution, ensuring models do not become biased towards the majority class.

# Data Splitting & Scaling:

Split the dataset into training (70%) and testing (30%) sets, using stratify to maintain class proportions.

Applied StandardScaler to all numerical features, fitting only on the training data and transforming both training and test sets to standardize feature ranges.

# Model Development & Training:

# Logistic Regression:

Trained on the scaled training data (X_train_scaled_df, y_train).

Used class_weight='balanced' to account for the original class imbalance.

A custom probability threshold (e.g., 0.25) was applied to predictions to decrease False Negatives, prioritizing the identification of actual defaults due to their high cost.

# K-Nearest Neighbors (KNN):

Trained on the SMOTE-resampled and scaled training data (X_train_resampled_df, y_train_resampled).

# Model Evaluation:

Evaluated both models using a comprehensive set of metrics crucial for imbalanced classification:

Accuracy: Overall correct predictions.

Precision, Recall, F1-Score: Focused on the "Default" class (Status=1) to assess the models' ability to correctly identify defaults and avoid false alarms.

ROC AUC Score: Measures the overall discriminative power of the models.

Confusion Matrix: Visualizes the breakdown of true positives, true negatives, false positives, and false negatives.

# Comparitative Analysis
In our case, a lower accuracy is acceptable because our primary focus is on maximizing recall, not overall correctness. This is especially important in loan default prediction, where a false negative (predicting someone will repay when they actually default) can lead to significant financial losses for the lender. By lowering the classification threshold, we prioritize catching more actual defaulters, even if it means mistakenly flagging some safe applicants. Therefore, although accuracy may drop slightly, a higher recall ensures that the model is more conservative and effective in identifying risky borrowers — which aligns with the business goal of minimizing loan default risk.

On using probability threshold as 0.5 we can get accuracy of 83 % for logistic regression and 86% for KNN
Logistic Regression vs. KNN: While KNN might show a higher overall accuracy, Logistic Regression often performs competitively or even better in scenarios where the underlying relationships are predominantly linear or can be approximated as such after extensive preprocessing.

Linear Decision Boundaries: Logistic Regression is inherently a linear model, meaning it tries to find a linear decision boundary to separate the classes. Given the extensive data preprocessing, including transformations (like log1p and sqrt) which aim to linearize relationships and normalize distributions, the data may have become more amenable to a linear separation.

Interpretability: Logistic Regression provides clear feature coefficients, indicating the strength and direction of each feature's influence on the log-odds of default. This offers greater interpretability into why a loan might default, which is highly valuable for business decisions. In a financial context, understanding the drivers of default (e.g., "higher income decreases default risk") is often as important as the prediction itself.

Robustness to Noise/Outliers (Post-Preprocessing): Although KNN can capture complex non-linear relationships, it is also sensitive to noise and outliers. While SMOTE helps balance the training data, and scaling normalizes features, the inherent instance-based nature of KNN can still be influenced by noisy data points. Logistic Regression, with its global optimization approach, can sometimes be more robust after careful preprocessing.

Computational Efficiency: For very large datasets, Logistic Regression is generally more computationally efficient during prediction than KNN, as KNN requires calculating distances to all training points for each new prediction.

# Technologies Used
Python: Programming Language
Pandas: Data manipulation and analysis
NumPy: Numerical operations
Scikit-learn: Machine learning models (Logistic Regression, KNN), data splitting, scaling, and evaluation metrics
Imbalanced-learn (imblearn): For handling class imbalance (SMOTE)
Matplotlib & Seaborn: Data visualization
