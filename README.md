# 🧹 Data Cleanser

## 📌 Project Overview

**Data Cleanser** is a Python-based data preprocessing and data cleaning project focused on transforming raw and inconsistent healthcare data into a clean, reliable, and analysis-ready dataset.

The project uses a **patient health records dataset containing 1,000 records and 9 columns**, including patient information, demographic attributes, health measurements, and disease-risk information.
The notebook demonstrates different techniques for handling missing values, imputing numerical and categorical variables, creating missing-value indicators, and preparing the dataset for further analysis.

---

## 🎯 Project Objective

The main objective of this project is to clean, preprocess, and transform raw and inconsistent healthcare data into a **high-quality, structured, reliable, and analysis-ready dataset**.

### Key Objectives

* 🧹 Identify and handle missing values.
* 📊 Analyze the distribution and characteristics of the dataset.
* 🔄 Apply appropriate missing-value imputation techniques.
* 📝 Handle categorical missing values using the most frequent strategy.
* 📐 Apply numerical imputation techniques.
* 🤖 Use KNN-based imputation for numerical variables.
* 🔬 Use MICE / Iterative Imputation for numerical variables.
* 🚩 Identify and treat outliers.
* ✅ Improve overall data quality and consistency.
* 📈 Prepare the cleaned dataset for further statistical analysis and machine learning.

---

## 📂 Dataset Information

The dataset contains **1,000 patient records and 9 columns**.

### Dataset Columns

| Column           | Description                |
| ---------------- | -------------------------- |
| `patient_id`     | Unique patient identifier  |
| `age`            | Age of the patient         |
| `gender`         | Gender of the patient      |
| `region`         | Patient region             |
| `bmi`            | Body Mass Index            |
| `blood_pressure` | Blood pressure measurement |
| `cholesterol`    | Cholesterol level          |
| `glucose`        | Glucose level              |
| `disease_risk`   | Disease-risk indicator     |

The original dataset contains missing values in `age`, `gender`, `region`, `bmi`, `cholesterol`, and `glucose`.

---

## 🛠️ Technologies & Libraries

The project is implemented using Python and the following libraries:

```python
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import scipy.stats as stats
from scipy.stats.mstats import winsorize
```

---

## 🔍 Project Workflow

```text
Raw Dataset
     ↓
Load Dataset
     ↓
Explore Dataset
     ↓
Check Shape & Data Types
     ↓
Identify Missing Values
     ↓
Missing Value Analysis
     ↓
Apply Imputation Techniques
     ↓
Validate Missing Values
     ↓
Outlier Detection & Treatment
     ↓
Final Clean Dataset
     ↓
Ready for Analysis / Machine Learning
```

---

# 📊 Part A — Missing Value Treatment

## 1. Load Dataset

```python
DF = pd.read_csv("patient_health_records.csv")

DF_imput = DF.copy()
```

The original dataset is preserved in `DF`, while `DF_imput` is used for preprocessing.

---

## 2. Check Dataset Shape

```python
print("Rows: ", DF.shape[0])
print("Columns: ", DF.shape[1])
print("Shape: ", DF.shape)
```

### Dataset Size

```text
Rows: 1000
Columns: 9
Shape: (1000, 9)
```

---

## 3. Check Data Information

```python
DF.info()
```

This helps identify:

* Number of records
* Column names
* Missing values
* Data types
* Memory usage

---

## 4. View First 10 Records

```python
DF.head(10)
```

---

## 5. Statistical Summary

```python
DF.describe()
```

This provides statistical information such as:

* Count
* Mean
* Standard deviation
* Minimum
* 25th percentile
* Median
* 75th percentile
* Maximum

---

## 6. Missing Value Percentage

```python
Missing_Report = (DF.isnull().sum() / len(DF)) * 100

print("--- Missing Values Summary (%) ---")
print(Missing_Report[Missing_Report > 0])
```

The notebook identifies missing values in:

```text
age            10.0%
gender         10.0%
region         10.0%
bmi             9.8%
cholesterol     9.7%
glucose         9.8%
```

---

# 🧮 Missing Value Imputation Techniques

## 7. Mean Imputation — BMI

For the numerical `bmi` column, mean imputation is applied.

```python
BMI_Imputer = SimpleImputer(strategy='mean')

DF_imput['bmi'] = BMI_Imputer.fit_transform(
    DF_imput[['bmi']]
)
```

### Strategy

```text
strategy = 'mean'
```

The missing BMI values are replaced with the mean BMI.

---

## 8. Most Frequent Imputation — Region

For the categorical `region` column, the **most frequent strategy** is used.

```python
Region_imputer = SimpleImputer(
    strategy='most_frequent'
)

DF_imput['region'] = Region_imputer.fit_transform(
    DF_imput[['region']]
).ravel()
```

### Strategy

```text
strategy = 'most_frequent'
```

The missing values are replaced by the most frequently occurring region.

---

## 9. Most Frequent Imputation — Gender

The same **Most Frequent / Mode Strategy** is applied to the `gender` column.

```python
Gender_imputer = SimpleImputer(
    strategy='most_frequent'
)

DF_imput[['gender']] = Gender_imputer.fit_transform(
    DF_imput[['gender']]
)
```

This is equivalent to:

```python
DF_imput['gender'] = DF_imput['gender'].fillna(
    DF_imput['gender'].mode()[0]
)
```

---

# 🎲 10. Random Sample Imputation — Age

A missing-value indicator is first created for the `age` column.

```python
DF_imput['age_missing_indicator'] = (
    DF_imput['age'].isnull().astype(int)
)
```

The available age values are extracted:

```python
age_non_missing = DF_imput['age'].dropna()

missing_mask = DF_imput['age'].isnull()
```

Random values are then selected from the observed age values:

```python
random_samples = np.random.choice(
    age_non_missing,
    size=missing_mask.sum()
)

DF_imput.loc[missing_mask, 'age'] = random_samples
```

This preserves the existing distribution of observed age values while filling the missing records.

---

# 🤖 11. KNN Imputation

K-Nearest Neighbors imputation is used for numerical variables.

```python
Num_cols = ['cholesterol', 'glucose']
```

Create the KNN imputer:

```python
knn_imputer = KNNImputer(
    n_neighbors=5
)
```

Apply KNN imputation:

```python
DF_knn_imputed = pd.DataFrame(
    knn_imputer.fit_transform(
        DF_imput[Num_cols]
    ),
    columns=Num_cols
)
```

Store the imputed cholesterol values:

```python
DF_imput['cholesterol_knn'] = (
    DF_knn_imputed['cholesterol']
)
```

The notebook uses:

```text
n_neighbors = 5
```

---

# 🔬 12. MICE / Iterative Imputation

Iterative Imputer is used for numerical missing values.

```python
mice_imputer = IterativeImputer(
    random_state=42
)
```

Apply the imputer:

```python
DF_mice_imputed = pd.DataFrame(
    mice_imputer.fit_transform(
        DF_imput[Num_cols]
    ),
    columns=Num_cols
)
```

Store the imputed glucose values:

```python
DF_imput['glucose_mice'] = (
    DF_mice_imputed['glucose']
)
```

---

# 🔄 13. Replace Original Columns

After comparing the imputation results, the generated columns are used to replace the original missing-value columns.

```python
DF_imput['cholesterol'] = (
    DF_imput['cholesterol_knn']
)

DF_imput['glucose'] = (
    DF_imput['glucose_mice']
)
```

Temporary columns are removed:

```python
DF_imput.drop(
    ['cholesterol_knn', 'glucose_mice'],
    axis=1,
    inplace=True
)
```

---

# ✅ 14. Verify Missing Values

```python
DF_imput.isnull().sum()
```

### Final Result

```text
patient_id               0
age                      0
gender                   0
region                   0
bmi                      0
blood_pressure           0
cholesterol              0
glucose                  0
disease_risk             0
age_missing_indicator    0
```

The notebook shows that all missing values have been handled after the imputation stage.

---

# 🚩 Part B — Outlier Treatment

The notebook then creates a separate copy of the imputed dataset for outlier analysis.

```python
DF_outliers = DF_imput.copy()
```

The project uses statistical techniques to identify and handle extreme observations.

The following library is imported for winsorization:

```python
from scipy.stats.mstats import winsorize
```

---

# 📈 Data Cleaning Techniques Used

| Technique                   | Column / Purpose        |
| --------------------------- | ----------------------- |
| Mean Imputation             | `bmi`                   |
| Most Frequent Imputation    | `gender`, `region`      |
| Random Sample Imputation    | `age`                   |
| Missing Indicator           | `age`                   |
| KNN Imputation              | `cholesterol`           |
| MICE / Iterative Imputation | `glucose`               |
| Outlier Treatment           | Numerical variables     |
| Winsorization               | Extreme-value treatment |

---

# 📦 Project Structure

```text
Data-Cleanser/
│
├── 📓 Data Cleanser.ipynb
├── 📄 patient_health_records.csv
├── 📄 README.md
└── 🎥 Project Video
```

---

# 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/TrivediMan/Data-Cleanser.git
```

### 2. Navigate to the Project Folder

```bash
cd Data-Cleanser
```

### 3. Install Required Libraries

```bash
pip install pandas numpy seaborn matplotlib scikit-learn scipy
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
Data Cleanser.ipynb
```

### 6. Run the Notebook

Execute the cells sequentially to reproduce the complete data-cleaning workflow.

---

# 🎥 Project Video

> **Video Demonstration:**
> Replace the URL below with your project video link.

🔗 **[▶️ Watch Project Video](YOUR_VIDEO_LINK_HERE)**

```text
YOUR_VIDEO_LINK_HERE
```

---

# 💡 Key Learning Outcomes

Through this project, the following data-cleaning concepts were demonstrated:

* Understanding missing-data patterns
* Mean-based numerical imputation
* Mode / most-frequent categorical imputation
* Random sample imputation
* Missing-value indicators
* KNN-based imputation
* MICE / Iterative imputation
* Data validation after preprocessing
* Outlier detection and treatment
* Preparing datasets for machine-learning workflows

---

# 📊 Final Outcome

The project transforms the original patient dataset from a dataset containing missing values into a cleaned dataset with **no remaining missing values**, while retaining a missingness indicator for the original `age` field.

The resulting dataset is suitable for:

* 📈 Exploratory Data Analysis
* 📊 Statistical Analysis
* 🤖 Machine Learning
* 🔍 Predictive Modeling
* 📉 Data Visualization

---

# 👨‍💻 Author

**Man Trivedi**

Data Analyst & Business Intelligence Specialist

🔗 GitHub: https://github.com/TrivediMan

🔗 LinkedIn: https://www.linkedin.com/in/man-trivedi-1bb663372

---

## ⭐ If You Find This Project Useful

If this project helped you understand data cleaning and missing-value treatment, consider giving the repository a ⭐ **Star** on GitHub.
