# CreditPathAI 🏦

**Automating and optimizing the loan recovery lifecycle by modeling repayment behavior using diverse data**

### 👨‍💻 Developed by: **Rajath Raj K T**
*Springboard Infosys Virtual Internship Program*

---

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Academic Information](#-academic-information)
- [Objectives](#-objectives)
- [Dataset Features](#-dataset-features)
- [Dataset Sources](#-dataset-sources)
- [Project Structure](#-project-structure)
- [Technologies Used](#️-technologies-used)
- [Installation & Setup](#️-installation--setup)
- [Using the Application](#-using-the-application)
- [Business Impact](#-business-impact)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights)
- [Project Workflow](#-project-workflow)
- [Future Enhancements](#-future-enhancements)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact--support)

## 📋 Project Overview

CreditPathAI is an end-to-end machine learning project that leverages advanced data analytics to predict loan defaults and optimize the loan recovery process. The project features a **complete interactive web application** built with Streamlit, allowing users to predict loan default risk in real-time using multiple machine learning models. By analyzing diverse customer data including demographics, financial history, and loan characteristics, this system helps financial institutions make informed lending decisions and reduce credit risk.

## 🎓 Academic Information

- **Developer**: **Rajath Raj K T** 👨‍💻
- **Program**: Springboard Infosys Virtual Internship
- **Mentor**: Dr. N Jagan Mohan
- **Organization**: Springboard Infosys
- **Year**: 2025

## 🎯 Objectives

- **Predict loan defaults** with high accuracy using 7 different machine learning models
- **Interactive web application** for real-time loan default risk assessment
- **Automate risk assessment** in the lending process
- **Compare model performance** with comprehensive metrics (Precision, Recall, F1-Score)
- **Optimize loan recovery strategies** based on customer behavior patterns
- **Reduce financial losses** by identifying high-risk borrowers early
- **Improve decision-making** through data-driven insights and visual analytics

## 📊 Dataset Features

The project uses a comprehensive loan dataset with **24+ key features**:

### Customer Demographics
- Gender, Age Bracket
- Region (North, South, Central, North-East)

### Financial Information
- Credit Score, Credit Worthiness
- Debt-to-Income Ratio (DTI)
- Annual Income
- Credit Type (EQUI, CRIF, CIB, EXP)

### Loan Details
- Loan Amount, Interest Rate, Loan Term
- Loan Purpose, Loan Type
- Loan Limit (Conforming/Non-Conforming)
- Pre-approval Status
- Loan-to-Value Ratio (LTV)

### Property & Collateral
- Property Value
- Occupancy Type (Primary, Secondary, Investment)
- Total Units
- Business or Commercial Property

### Loan Structure
- Negative Amortization
- Interest Only
- Lump Sum Payment
- Submission of Application

### Target Variable
- **Status**: Binary classification (0 = No Default, 1 = Default)

## � Dataset Sources

This project leverages **two comprehensive datasets** for robust loan default prediction:

### 1. Kaggle Loan Default Dataset
- **Source**: [Loan Default Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
- **Description**: Contains detailed loan application data with 24+ features including borrower demographics, financial metrics, and loan characteristics
- **Usage**: Primary dataset for model training and validation

### 2. Microsoft Loan Credit Risk Dataset
- **Source**: [Microsoft R Server Loan Credit Risk](https://microsoft.github.io/r-server-loan-credit-risk/dba.html)
- **Description**: Enterprise-grade dataset with borrower and loan information used for credit risk analysis
- **Usage**: Additional validation and comparative analysis

Both datasets provide complementary perspectives on loan default patterns, enabling more robust model development and validation.

## �🚀 Project Structure

```
CreditPathAI/
├── streamlit_app/              # Interactive Web Application
│   ├── app.py                  # Main Streamlit application
│   ├── utils.py                # Utility functions for model loading & prediction
│   ├── requirements.txt        # App-specific dependencies
│   ├── models/                 # Trained model pipelines (generated locally)
│   │   ├── logistic_regression_pipeline.joblib
│   │   ├── random_forest_pipeline.joblib
│   │   ├── xgboost_pipeline.joblib
│   │   ├── decision_tree_pipeline.joblib
│   │   ├── k-nearest_neighbors_pipeline.joblib
│   │   ├── gaussian_naive_bayes_pipeline.joblib
│   │   └── bernoulli_naive_bayes_pipeline.joblib
│   └── __pycache__/            # Python cache files
│
├── notebooks/                  # Jupyter Notebooks
│   ├── eda_report.ipynb       # Exploratory Data Analysis (Kaggle dataset)
│   ├── loan_default_main.ipynb            # Main ML pipeline development
│   ├── pre_processing__methods_2.ipynb    # Data preprocessing experiments
│   └── pre_processing__methods_2_updated.ipynb  # Final preprocessing & model training
│
├── microsoft_notebooks/        # Microsoft Dataset Analysis
│   ├── eda_report.ipynb       # EDA for Microsoft dataset
│   └── microsoft_loan_default.ipynb       # Microsoft dataset modeling
│
├── Loan_Default.csv           # Primary dataset (Kaggle)
├── Loan.txt                   # Loan data (Microsoft dataset)
├── Loan_Prod.txt              # Production loan data
├── Borrower.txt               # Borrower data (Microsoft dataset)
├── Borrower_Prod.txt          # Production borrower data
├── Model_comparison.xlsx      # Model performance comparison results
│
├── requirements.txt           # Project dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
└── LICENSE                    # MIT License
```

**Note**: Model files (`*.joblib`) are excluded from version control due to GitHub's file size limits. Users need to train models locally using the provided notebooks.

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.13** - Primary programming language
- **Streamlit** - Interactive web application framework

### Data Science & Machine Learning
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn 1.7.2** - Machine learning algorithms and preprocessing
- **XGBoost** - Gradient boosting framework

### Machine Learning Models
1. **Logistic Regression** - Baseline linear classifier
2. **Random Forest** - Ensemble learning with decision trees
3. **XGBoost** - Gradient boosting classifier
4. **Decision Tree** - Single tree classifier
5. **K-Nearest Neighbors (KNN)** - Instance-based learning
6. **Gaussian Naive Bayes** - Probabilistic classifier
7. **Bernoulli Naive Bayes** - Binary feature classifier

### Data Visualization
- **Matplotlib** - Static data visualization
- **Seaborn** - Statistical graphics

### Development Tools
- **Jupyter Notebook** - Interactive development environment
- **Joblib** - Model serialization and persistence
- **Git** - Version control

### Deployment
- **Virtual Environment (.venv)** - Isolated Python environment
- **Git LFS** - Large file storage (for model files)

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.13 or higher
- Git
- Virtual environment support

### Step 1: Clone the Repository
```bash
git clone https://github.com/springboardmentor891v/CreditPathAI.git
cd CreditPathAI
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install main project dependencies
pip install -r requirements.txt

# Install Streamlit app dependencies
pip install -r streamlit_app/requirements.txt
```

### Step 4: Train Models
Since model files are not included in the repository (due to size constraints), you need to train them locally:

1. Open the training notebook:
   ```bash
   jupyter notebook notebooks/pre_processing__methods_2_updated.ipynb
   ```

2. Run all cells in the notebook, especially the final cell that trains and saves all 7 model pipelines

3. Verify that `.joblib` files are created in `streamlit_app/models/` directory

### Step 5: Launch the Streamlit App
```bash
streamlit run streamlit_app/app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎮 Using the Application

### Features
1. **Model Selection** - Choose from 7 different ML models
2. **Interactive Input Form** - Enter loan applicant details via intuitive sliders and dropdowns
3. **Real-time Prediction** - Get instant default risk predictions
4. **Risk Assessment** - Color-coded risk levels (Low/Medium/High)
5. **Model Performance Metrics** - View Precision, Recall, and F1-Score for each model
6. **Comparative Analysis** - Compare performance across all models

### Input Parameters
The app accepts 24 different features including:
- **Demographics**: Gender, Age, Region
- **Financial**: Credit Score, Income, DTI Ratio
- **Loan Details**: Amount, Interest Rate, Term, Purpose
- **Property**: Value, Occupancy Type, Total Units
- And more...

### Output
- **Prediction**: Default or No Default
- **Probability**: Confidence score (0-100%)
- **Risk Level**: Low (🟢), Medium (🟡), or High (🔴) risk assessment



## 🎯 Business Impact

- **Risk Reduction**: Early identification of potential defaulters with 7 different model approaches
- **Cost Optimization**: Reduced loan recovery costs through targeted strategies and accurate predictions
- **Revenue Enhancement**: Better loan approval decisions leading to improved profitability
- **Process Automation**: Streamlined lending workflow with AI-driven insights via web interface
- **Real-time Decision Making**: Instant risk assessment through interactive web application
- **Model Flexibility**: Choose the best model based on specific business requirements (precision vs recall)
- **Scalability**: Web-based deployment ready for enterprise integration

## 📊 Model Performance

The project implements and compares 7 machine learning models:

| Model | Use Case | Key Strength |
|-------|----------|--------------|
| **Logistic Regression** | Baseline & interpretability | Simple, fast, interpretable coefficients |
| **Random Forest** | High accuracy | Ensemble learning, feature importance |
| **XGBoost** | Best performance | Advanced boosting, handles imbalanced data |
| **Decision Tree** | Rule extraction | Easy to visualize and explain |
| **K-Nearest Neighbors** | Pattern matching | Instance-based learning |
| **Gaussian Naive Bayes** | Fast prediction | Probabilistic, works well with continuous features |
| **Bernoulli Naive Bayes** | Binary features | Efficient for categorical data |

All models are evaluated using:
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **Class Balancing**: Handled through `class_weight='balanced'` and `scale_pos_weight`

## 💡 Key Insights

- Analysis of customer demographics impact on default rates across two different datasets
- Correlation between credit scores and repayment behavior patterns
- Impact of loan-to-value ratio (LTV) and debt-to-income ratio (DTI) on default risk
- Regional variations in loan default patterns
- Optimal loan terms and interest rates for different customer segments
- Pre-approval status as a significant predictor
- Property occupancy type correlation with default probability
- Comparative analysis between Kaggle and Microsoft datasets for model validation

- Comparative analysis between Kaggle and Microsoft datasets for model validation

## 🔄 Project Workflow

### 1. Data Collection & Exploration
- Import datasets from Kaggle and Microsoft sources
- Exploratory Data Analysis (EDA) in dedicated notebooks
- Feature analysis and statistical summaries

### 2. Data Preprocessing
- Handle missing values using appropriate imputation strategies
- Encode categorical variables (One-Hot Encoding)
- Scale numerical features (StandardScaler)
- Address class imbalance issues

### 3. Feature Engineering
- Create preprocessing pipelines using `ColumnTransformer`
- Separate numerical and categorical transformers
- Ensure consistent preprocessing across train/test splits

### 4. Model Training
- Train 7 different classification models
- Use stratified train-test split
- Apply class balancing techniques
- Hyperparameter optimization for key models

### 5. Model Evaluation
- Compare models using Precision, Recall, and F1-Score
- Generate performance comparison reports
- Save trained model pipelines using Joblib

### 6. Deployment
- Load trained models in Streamlit application
- Create interactive UI for real-time predictions
- Implement model selection and comparison features

## 🚀 Future Enhancements

- [ ] **Advanced Feature Engineering**: Derive new features from existing data
- [ ] **Hyperparameter Tuning**: GridSearchCV/RandomizedSearchCV for all models
- [ ] **Deep Learning**: Experiment with neural networks
- [ ] **Model Explainability**: SHAP values and LIME for model interpretation
- [ ] **API Development**: RESTful API for programmatic access
- [ ] **Database Integration**: Connect to production databases
- [ ] **A/B Testing**: Compare model performance in production
- [ ] **Monitoring Dashboard**: Track model performance over time
- [ ] **Auto-retraining Pipeline**: Automated model updates with new data

## 🤝 Contributing

This project is part of the Springboard Infosys Virtual Internship program. 

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss proposed changes.

For major changes, please open an issue first to discuss proposed changes.

## � Troubleshooting

### Common Issues

**1. Model files not found error**
```
Solution: Run the training notebook (pre_processing__methods_2_updated.ipynb) 
to generate model files locally. Model files are excluded from Git due to size.
```

**2. Package version conflicts**
```
Solution: Use the exact versions specified in requirements.txt:
- scikit-learn==1.7.2
- xgboost (latest compatible version)
```

**3. Streamlit app won't start**
```
Solution: Ensure virtual environment is activated and all dependencies are installed:
pip install -r streamlit_app/requirements.txt
```

**4. Import errors in notebooks**
```
Solution: Install all project dependencies:
pip install -r requirements.txt
```

## 📞 Contact & Support

### Developer
**Rajath Raj K T** 👨‍💻  
Springboard Infosys Virtual Intern

### Project Information
- **Mentor**: Dr. N Jagan Mohan
- **Organization**: Springboard Infosys
- **Repository**: [GitHub - CreditPathAI](https://github.com/springboardmentor891v/CreditPathAI)

For questions, issues, or suggestions, please open an issue on GitHub.

## �📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍🏫 Acknowledgments

- **Rajath Raj K T** - Lead Developer & Machine Learning Engineer
- **Dr. N Jagan Mohan** - Project mentor for guidance and technical support
- **Springboard Infosys** - For providing the internship opportunity and resources
- **Kaggle Community** - For the comprehensive loan default dataset
- **Microsoft** - For the enterprise-grade credit risk dataset
- **Open Source Community** - For the amazing tools and libraries (Scikit-learn, XGBoost, Streamlit)

## 📈 Project Stats

- **Total Models**: 7 Machine Learning Algorithms
- **Dataset Features**: 24+ loan and borrower attributes
- **Data Sources**: 2 (Kaggle + Microsoft)
- **Notebooks**: 6 comprehensive analysis and training notebooks
- **Lines of Code**: 2000+ (excluding libraries)
- **Model Accuracy**: Varies by model (see performance metrics in app)

---

**⭐ If you find this project helpful, please consider giving it a star on GitHub!**

**💼 Developed by Rajath Raj K T | Springboard Infosys Virtual Internship 2025**

**Note**: This project is developed as part of an educational internship program and is intended for learning and demonstration purposes. It showcases end-to-end machine learning workflow from data exploration to deployment.
