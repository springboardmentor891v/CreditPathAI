# Understanding Machine Learning (ML)

## 1. Introduction

Machine Learning is a branch of Artificial Intelligence that enables systems to **learn from data** and **make predictions or decisions** without being explicitly programmed.  

**Goal:** Improve performance on tasks automatically through experience.  

**Example:** Spam email detection, recommendation systems, stock price prediction.  

---

## 2. Key ML Types

### A. Supervised Learning
Learning from labeled data. Predict an output based on input features.  

- **Regression:** Predict continuous values (e.g., house prices).  
- **Classification:** Predict categories (e.g., email spam or not).  

### B. Unsupervised Learning
Learning from unlabeled data. Discover hidden patterns or structures.  

- **Clustering:** Group similar data (e.g., customer segmentation).  
- **Dimensionality Reduction:** Reduce features for easier analysis (e.g., PCA).  

### C. Reinforcement Learning
Learning via trial and error with rewards.  

- **Agent** interacts with an environment and learns from feedback.  
- **Example:** Training a robot to walk, game AI.  

---

## 3. Familiarity with Foundational ML Algorithms

| Type          | Algorithm                          | Use Case                                                |
| ------------- | ---------------------------------- | ------------------------------------------------------- |
| Supervised    | Linear Regression                  | Predict house prices, sales forecasting                 |
| Supervised    | Logistic Regression                | Email spam detection, customer churn prediction         |
| Supervised    | Decision Trees                     | Loan approval, classification tasks                     |
| Supervised    | Random Forest                      | Improves decision tree predictions, reduces overfitting |
| Unsupervised  | K-Means Clustering                 | Customer segmentation, image compression                |
| Unsupervised  | PCA (Principal Component Analysis) | Dimensionality reduction for visualization              |
| Reinforcement | Q-Learning                         | Game AI, robot navigation                               |

---

# Linear Regression Documentation

## 1. Introduction

Linear Regression is a **supervised machine learning algorithm** used for predicting a **continuous dependent variable (target)** based on one or more **independent variables (features)**.  

**Goal:** Find a relationship between input features and output to make predictions.  

**Equation (Simple Linear Regression):**  

\[
y = mx + c
\]  

Where:  
- `y` = predicted value (dependent variable)  
- `x` = input feature (independent variable)  
- `m` = slope (coefficient)  
- `c` = intercept  

---

## 2. Types of Linear Regression

1. **Simple Linear Regression** – One independent variable predicts the dependent variable.  
2. **Multiple Linear Regression** – Two or more independent variables predict the dependent variable.  

\[
y = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_n x_n
\]  

---

## 3. Assumptions of Linear Regression

Linear Regression works well if these assumptions are met:  

1. Linear relationship between dependent and independent variables.  
2. No or little multicollinearity among features.  
3. Homoscedasticity – constant variance of residuals.  
4. Residuals are normally distributed.  

---

## 4. Use Cases

- Predicting house prices, salaries, or sales.  
- Forecasting stock prices.  
- Estimating product demand based on features like advertising spend, location, or size.  

---

## 5. Advantages

- Simple to understand and interpret.  
- Efficient to train on small and medium-sized datasets.  
- Can be extended to multiple variables.  

---

## 6. Limitations

- Assumes linear relationship (not suitable for non-linear data).  
- Sensitive to outliers.  
- Poor performance if features are highly correlated.  


## 1. Need for Train-Test-Split
- Machine learning models should **generalize** to unseen data, not just memorize training data.  
- If we train and test on the same dataset, results may look very accurate but will fail on new data (overfitting).  
- **Train-Test Split** divides data into:
  - **Training Set** → used to learn patterns.
  - **Test Set** → used to evaluate on unseen borrowers.  

**Loan Default Example:**  
- Train set → learn patterns like income, loan amount, credit history affecting default.  
- Test set → check accuracy on unseen borrowers.  

---

## 2. Understanding StandardScaler in scikit-learn
- Features often have different ranges (e.g., `Loan_Amount` in thousands vs `Credit_History` as 0/1).  
- Some algorithms (Logistic Regression, KNN, SVM) are **sensitive to feature scale**.  
- **StandardScaler** standardizes features:  

\[
z = \frac{x - \mu}{\sigma}
\]  

- After scaling: mean = 0, standard deviation = 1.  
- Ensures all features contribute equally to the model.  

**Loan Default Example:**  
- Without scaling → loan amount dominates because of large values.  
- With scaling → income, loan amount, credit history treated fairly.  

---

## 3. Difference between Scaling and Normalization
- **Scaling (Standardization):**
  - Transforms data to mean = 0, std = 1.
  - Good when features have different units.  
  - Example: `Loan Amount`, `Applicant Income`.  

- **Normalization (Min-Max Scaling):**
  - Transforms values into range [0, 1].  
  - Useful for distance-based models (KNN, Neural Networks).  
  - Example: Loan amount = ₹50,000 → 0.5 (if min=0, max=100,000).  

**When to Use:**  
- **StandardScaler** → for models like Logistic Regression, SVM.  
- **Normalization** → for models where relative distances matter, like KNN, Neural Networks.  

---

## 4. Role of `random_state` and `stratify`
- **random_state:**
  - Ensures reproducibility.  
  - Using the same random_state gives the same train/test split every time.  

- **stratify:**
  - Keeps the **same class distribution** in train and test as in the original dataset.  
  - Very important in imbalanced datasets like Loan Default (where defaulters are fewer).  
  - Without stratify → test set may have fewer or no default cases → poor evaluation.  
  - With stratify → balanced distribution in both sets.  

---