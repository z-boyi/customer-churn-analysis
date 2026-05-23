# Customer Churn Analysis: Predictive Modeling & Business Insights

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive data science project analyzing customer churn using exploratory data analysis, statistical testing, and machine learning. This project demonstrates end-to-end ML pipeline development with production-ready code architecture.

## 📊 Key Highlights

- **Dataset**: 7,043 customers with 20+ behavioral features
- **Churn Rate**: 26.5% (highly actionable class distribution)
- **Best Model**: Gradient Boosting Classifier
- **Test ROC-AUC**: 0.85+ (hyperparameter tuned)
- **Models Compared**: 3 (Logistic Regression, Random Forest, Gradient Boosting)
- **Cross-Validation**: 5-fold stratified with multiple metrics

---

## 🎯 Project Objectives

1. **Understand Churn Drivers**: Identify key factors influencing customer churn
2. **Predictive Modeling**: Build interpretable and high-performing classification models
3. **Business Recommendations**: Translate data insights into actionable strategies
4. **Production-Ready Code**: Demonstrate professional ML engineering practices

---

## 📁 Project Structure

```
customer-churn-analysis/
├── data/
│   ├── raw_churn.csv                 # Original dataset
│   └── clean_churn.csv               # Preprocessed dataset
│
├── src/                              # Reusable Python modules
│   ├── __init__.py
│   ├── config.py                     # Configuration & constants
│   ├── data.py                       # Data loading & preprocessing
│   ├── features.py                   # Feature engineering & encoding
│   ├── modeling.py                   # Model training & evaluation
│   └── visualization.py              # Plotting utilities
│
├── notebooks/                        # Jupyter notebooks (exploratory)
│   ├── 01_load_and_inspect.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_statistical_analysis.ipynb
│   └── 05_modeling.ipynb
│
├── models/                           # Trained models & results
│   ├── *.pkl                         # Serialized models
│   ├── *_metadata.json               # Model metadata
│   └── *.csv                         # Evaluation results
│
├── figures/                          # EDA & model visualizations
│
├── train_and_evaluate.py             # Advanced training pipeline
├── predict.py                        # Inference & batch predictions
├── requirements.txt                  # Python dependencies
├── Makefile                          # Pipeline automation
├── .gitignore
└── README.md
```

---

## 🔬 Methodology

### 1. Data Cleaning & Preprocessing
- Handled missing values in `TotalCharges` (filled with median)
- Removed non-informative identifiers (`customerID`)
- Engineered tenure-based features (categorical bucketing)
- Standardized categorical variable formatting

### 2. Exploratory Data Analysis (EDA)
- **Churn Distribution**: Analyzed class balance and patterns
- **Categorical Analysis**: Chi-square tests for independence
- **Numerical Analysis**: T-tests for monthly/total charges
- **Feature Relationships**: Correlation analysis & visual exploration
- **Key Findings**: 
  - Month-to-month contracts → 42% churn rate
  - New customers (0-12 months) → 48% churn rate
  - Electronic check payment → 45% churn rate

### 3. Statistical Analysis
- **Chi-Square Tests**: Categorical associations with churn
- **T-Tests**: Numerical feature differences by churn status
- **Logistic Regression**: Coefficient interpretation for churn drivers

### 4. Advanced Modeling Pipeline
- **Feature Engineering**: Label encoding (categorical) + standardization (numerical)
- **Hyperparameter Tuning**: GridSearchCV with 5-fold stratified CV
- **Model Comparison**: Evaluated 3 models on identical train/test splits
- **Performance Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Cross-Validation**: Mean ± std across folds

---

## 🏆 Model Performance

### Test Set Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------| 
| **Gradient Boosting** | **0.805** | **0.652** | **0.494** | **0.560** | **0.850** |
| Random Forest | 0.798 | 0.638 | 0.479 | 0.548 | 0.843 |
| Logistic Regression | 0.801 | 0.644 | 0.476 | 0.543 | 0.837 |

### Best Model (Gradient Boosting)
- **Cross-Validation ROC-AUC**: 0.847 ± 0.018
- **Key Hyperparameters**: n_estimators=100, learning_rate=0.05, max_depth=5
- **Best Features**: 
  - Contract type (month-to-month: -2.5 impact)
  - Tenure (per year increase: +0.12 impact)
  - Monthly charges (per $10 increase: +0.18 impact)
  - Tech support (has: +0.85 protection factor)

---

## 💡 Key Business Findings

| Factor | Impact | Insight |
|--------|--------|---------|
| **Contract Type** | 42% churn (M-t-M) vs 2.8% (2-year) | Contract length is strongest churn driver |
| **Tenure** | 48% churn (0-12 mo) vs 5% (24+ mo) | Critical onboarding period exists |
| **Payment Method** | 45% churn (e-check) vs 15% (auto-pay) | Payment friction increases churn |
| **Tech Support** | -27% churn reduction | Support services strongly protective |
| **Pricing** | +5% churn per $10 monthly | Price-sensitive segment exists |

---

## 🚀 Business Recommendations

1. **Incentivize Long-Term Contracts**
   - Offer 10-20% discount for 2-year commitment
   - Target new customers in first 6 months
   - Estimated impact: -10-15% churn reduction

2. **Implement Early Engagement Programs**
   - Onboarding calls in month 1-3
   - Technology training for new features
   - Estimated impact: -8-12% churn reduction

3. **Optimize Payment Experience**
   - Promote automatic payment (offer 5% discount)
   - Simplify billing for high-risk segments
   - Estimated impact: -5-8% churn reduction

4. **Strategic Service Bundling**
   - Bundle tech support with high-value plans
   - Cross-sell complementary services
   - Estimated impact: -10-15% churn reduction

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/z-boyi/customer-churn-analysis.git
cd customer-churn-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

#### Option 1: Run Full Pipeline
```bash
make pipeline
# Or: python train_and_evaluate.py
```

#### Option 2: Make Predictions
```bash
python predict.py
# Includes examples for single & batch predictions
```

#### Option 3: Explore Notebooks
```bash
jupyter notebook notebooks/
# Start with 01_load_and_inspect.ipynb
```

### Available Commands

```bash
make clean       # Remove cached files & models
make install     # Install dependencies
make pipeline    # Run training & evaluation
make predict     # Run inference examples
make help        # Show all commands
```

---

## 📈 Advanced Features

### Hyperparameter Tuning
- **GridSearchCV** with 5-fold stratified cross-validation
- Parameter grids for each model
- Results saved to `models/hyperparameter_tuning_results.json`

### Model Serialization
- Trained models saved as pickle files
- Metadata (feature names, training date) in JSON
- Reproducible inference pipeline

### Batch Predictions
- Process multiple customers efficiently
- Returns prediction + probability
- Identify high-risk customers (e.g., >40% churn probability)

### Cross-Validation
- Stratified splits to preserve class distribution
- Per-fold metrics (mean ± std)
- Assesses model generalization

---

## 📊 Reproducibility

All results are fully reproducible:

- **Random seeds**: Fixed at 42 for all stochastic operations
- **Train/Test Split**: 80/20 stratified split
- **Feature Preprocessing**: Saved encoders/scalers for consistency
- **Model Artifacts**: Serialized for deployment

---

## 📝 Files & Their Purpose

| File | Purpose |
|------|---------|
| `src/config.py` | Centralized configuration & paths |
| `src/data.py` | Data loading & cleaning logic |
| `src/features.py` | Feature encoding & scaling |
| `src/modeling.py` | Model training & evaluation utilities |
| `src/visualization.py` | Reusable plotting functions |
| `train_and_evaluate.py` | Full training pipeline |
| `predict.py` | Inference & batch prediction |

---

## 🔍 Limitations & Future Work

### Current Limitations
- No class imbalance handling (e.g., SMOTE, class weights)
- Limited temporal analysis (static snapshot)
- No interaction features or polynomial features
- Feature selection based on domain knowledge only

### Future Enhancements
- Implement SMOTE for imbalanced data
- Add temporal trends & seasonality
- Engineer interaction & polynomial features
- Use SHAP values for model explainability
- Deploy as REST API (Flask/FastAPI)
- Add automated retraining pipeline

---

## 👤 Author

**Data Science Engineer**  
*Passionate about turning data into actionable insights*

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙋 Questions & Feedback

For questions about this project, feel free to open an issue or reach out directly.

**Tags**: `machine-learning` `classification` `churn-prediction` `data-science` `scikit-learn` `pandas` `python`
