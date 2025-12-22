# Customer Churn Analysis

## Project Overview
This project analyzes customer churn behavior using a real-world telecommunications dataset.  
The goal is to identify key factors associated with customer churn and translate data insights into actionable business recommendations.

The project follows a structured data analytics workflow:
1. Data loading and inspection
2. Data cleaning and preprocessing
3. Exploratory data analysis (EDA)
4. Statistical analysis
5. Predictive modeling
6. Business insights and recommendations

---

## Dataset
- **Name**: Telco Customer Churn
- **Source**: IBM Sample Data / Kaggle
- **Size**: ~7,000 customer records
- **Description**:  
  The dataset contains customer demographics, service usage, contract details, and billing information, along with a binary churn indicator.

---

## Tools and Libraries
- Python
- pandas, numpy
- matplotlib, seaborn
- scipy, statsmodels
- scikit-learn
- Jupyter Notebook
- Git & GitHub

---

## Project Structure
```
.
├── data
│   ├── clean_churn.csv
│   └── raw_churn.csv
├── figures
├── notebooks
│   ├── 01_load_and_inspect.ipynb
│   └── 02_data_cleaning.ipynb
├── README.md
├── requirements.txt
└── src
```

---

## Workflow
1. **Data Loading & Inspection**
   - Initial data audit
   - Variable inspection and data type validation
   - Target variable exploration

2. **Data Cleaning**
   - Converted `TotalCharges` to numeric
   - Handled missing values
   - Encoded churn as a binary variable
   - Standardized categorical variables
   - Engineered tenure-based features

3. **Exploratory Data Analysis (next)**
   - Churn patterns by customer segments
   - Service and contract-level analysis
   - Visual insights

4. **Statistical Analysis & Modeling (upcoming)**
   - Hypothesis testing
   - Logistic regression
   - Model interpretation

---

## Key Data Cleaning Decisions
- Rows with invalid `TotalCharges` values were removed due to minimal impact (<0.2%)
- Customer identifiers were excluded from analysis
- Tenure was grouped to support customer segmentation

---

## Status
🚧 In progress  
- Data cleaning completed  
- EDA and modeling in progress