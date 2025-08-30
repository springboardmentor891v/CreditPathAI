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
