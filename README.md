# Customer Churn Analysis

## Project Overview
This project analyzes customer churn behavior using the Telco Customer Churn dataset.  
The goal is to identify key drivers of churn and provide actionable, data driven business insights.

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
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- scikit-learn
- Jupyter Notebook

---

## Project Structure
```
.
├── data
│   └── raw_churn.csv
├── figures
├── notebooks
│   └── 01_load_and_inspect.ipynb
├── README.md
├── requirements.txt
└── src
```


---

## Step 1: Data Loading and Inspection
**Notebook**: `notebooks/01_load_and_inspect.ipynb`

In this step, we:
- Load the raw dataset
- Inspect dimensions, column types, and structure
- Check for missing values and duplicates
- Examine the distribution of the target variable (`Churn`)
- Identify categorical and numerical features

### Key Initial Observations
- The dataset contains approximately 7,000 observations and 21 features
- The target variable `Churn` is binary and moderately imbalanced
- Most features are categorical
- `TotalCharges` is stored as a string and requires cleaning
- No duplicate records are present

This initial inspection informs all subsequent cleaning and analysis steps.

---

## Status
🚧 In progress  
Currently completed: **Step 1 – Data Loading and Inspection**

---

## Next Steps
- Clean and preprocess the data
- Perform exploratory data analysis
- Apply statistical tests to identify churn drivers
- Build interpretable predictive models
- Translate findings into business recommendations





