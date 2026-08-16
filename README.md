# Data Cleanser

## 📌 Project Overview

**Data Cleanser** is a Python-based data preprocessing project designed to clean and prepare a patient health dataset for further data analysis and machine learning.

The project focuses on:

* Loading a patient health dataset
* Understanding the structure of the dataset
* Detecting missing values
* Handling missing values using different imputation techniques
* Detecting and handling outliers
* Comparing data before and after outlier treatment
* Applying Winsorization
* Creating a final cleaned dataset
* Exporting the cleaned dataset to a CSV file

The project uses the dataset **`patient_health_records.csv`** and generates the cleaned dataset as **`output.csv`**.

---

## 📂 Project Structure

```text
Data-Cleanser/
│
├── Data Cleanser(2).ipynb
├── patient_health_records.csv
├── output.csv
└── README.md
```

---

## 🛠️ Technologies Used

The project is implemented using Python and the following libraries:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy

### Libraries Used

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

# 📊 Dataset

The input dataset is:

```text
patient_health_records.csv
```

The dataset contains **1000 rows and 9 columns**.

### Dataset Columns

| Column           | Description                |
| ---------------- | -------------------------- |
| `patient_id`     | Unique patient identifier  |
| `age`            | Patient age                |
| `gender`         | Patient gender             |
| `region`         | Patient region             |
| `bmi`            | Body Mass Index            |
| `blood_pressure` | Blood pressure measurement |
| `cholesterol`    | Cholesterol level          |
| `glucose`        | Glucose level              |
| `disease_risk`   | Disease risk indicator     |

The original dataset contains missing values in `age`, `gender`, `region`, `bmi`, `cholesterol`, and `glucose`.

---

# 🔍 1. Import Required Libraries

The first step is importing all libraries required for data manipulation, visualization, statistical analysis, and missing-value treatment.

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

# 📥 2. Load Dataset

The patient health records are loaded using Pandas.

```python
DF = pd.read_csv("patient_health_records.csv")
DF_imput = DF.copy()
```

A copy of the original dataset is created so that the original data remains unchanged during preprocessing.

---

# 📏 3. Dataset Shape

The number of rows and columns is checked.

```python
print("Rows: ", DF.shape[0])
print("Columns: ", DF.shape[1])
print("Shape: ", DF.shape)
```

### Output

```text
Rows:  1000
Columns:  9
Shape:  (1000, 9)
```

---

# 🔎 4. Dataset Information

The `info()` function is used to understand:

* Number of records
* Column names
* Missing values
* Data types
* Memory usage

```python
DF.info()
```

The dataset contains numerical and categorical columns. Missing values are present in several attributes.

---

# 👀 5. View First 10 Records

The first 10 records are displayed using:

```python
DF.head(10)
```

This helps to understand the structure and values of the dataset.

---

# 📈 6. Descriptive Statistics

Statistical information is obtained using:

```python
DF.describe()
```

This provides:

* Count
* Mean
* Standard deviation
* Minimum
* 25th percentile
* Median
* 75th percentile
* Maximum

The original dataset contains 1000 patient records.

---

# 🧹 Part A — Missing Value Treatment

## 7. Missing Value Percentage

The percentage of missing values in each column is calculated.

```python
Missing_Report = (DF.isnull().sum() / len(DF)) * 100

print("--- Missing Values Summary (%) ---")
print(Missing_Report[Missing_Report > 0])
```

### Missing Values Found

```text
age             10.0%
gender          10.0%
region          10.0%
bmi              9.8%
cholesterol      9.7%
glucose          9.8%
```

This shows that several columns require missing-value treatment.

---

# 8. BMI Missing Values — Mean Imputation

Missing BMI values are replaced using the mean of the BMI column.

```python
BMI_Imputer = SimpleImputer(strategy='mean')

DF_imput['bmi'] = BMI_Imputer.fit_transform(
    DF_imput[['bmi']]
)

DF_imput.isnull().sum()
```

### Method Used

```text
Mean Imputation
```

Mean imputation is useful for numerical variables when missing values need to be replaced with a representative central value.

---

# 9. Region Missing Values — Mode Imputation

Missing values in the `region` column are replaced using the most frequently occurring region.

```python
Region_imputer = SimpleImputer(strategy='most_frequent')

DF_imput['region'] = Region_imputer.fit_transform(
    DF_imput[['region']]
).ravel()

DF_imput.isnull().sum()
```

### Method Used

```text
Most Frequent / Mode Imputation
```

---

# 10. Gender Missing Values — Mode Imputation

Missing gender values are replaced using the mode.

```python
DF_imput['gender'] = DF_imput['gender'].fillna(
    DF_imput['gender'].mode()[0]
)
```

---

# 11. Age Missing Indicator

Before filling missing age values, an additional indicator column is created.

```python
DF_imput['age_missing_indicator'] = (
    DF_imput['age'].isnull().astype(int)
)
```

The new column indicates whether the original age value was missing.

```text
0 = Age was available
1 = Age was missing
```

---

# 12. Age — Random Sample Imputation

The non-missing age values are extracted.

```python
age_non_missing = DF_imput['age'].dropna()

missing_mask = DF_imput['age'].isnull()
```

Random samples are then selected from the available ages.

```python
random_samples = np.random.choice(
    age_non_missing,
    size=missing_mask.sum()
)

DF_imput.loc[missing_mask, 'age'] = random_samples
```

This preserves the general distribution of the observed age values instead of simply replacing all missing values with the mean.

---

# 13. Numerical Columns

The numerical columns requiring advanced imputation are selected.

```python
Num_cols = ['cholesterol', 'glucose']
```

---

# 14. KNN Imputation for Cholesterol

K-Nearest Neighbors imputation is applied to the selected numerical columns.

```python
knn_imputer = KNNImputer(n_neighbors=5)

DF_knn_imputed = pd.DataFrame(
    knn_imputer.fit_transform(DF_imput[Num_cols]),
    columns=Num_cols
)

DF_imput['cholesterol_knn'] = (
    DF_knn_imputed['cholesterol']
)
```

The imputed cholesterol values are temporarily stored in:

```text
cholesterol_knn
```

---

# 15. Iterative / MICE Imputation for Glucose

Iterative Imputation is used for glucose.

```python
mice_imputer = IterativeImputer(random_state=42)

DF_mice_imputed = pd.DataFrame(
    mice_imputer.fit_transform(DF_imput[Num_cols]),
    columns=Num_cols
)

DF_imput['glucose_mice'] = (
    DF_mice_imputed['glucose']
)
```

The notebook refers to this approach as **MICE (Iterative Imputer)**.

---

# 16. Replace Original Values

The imputed values are copied back to the original columns.

```python
DF_imput['cholesterol'] = DF_imput['cholesterol_knn']

DF_imput['glucose'] = DF_imput['glucose_mice']
```

Temporary columns are then removed.

```python
DF_imput.drop(
    ['cholesterol_knn', 'glucose_mice'],
    axis=1,
    inplace=True
)
```

---

# 17. Verify Missing Values

The final missing-value count is checked.

```python
DF_imput.isnull().sum()
```

The notebook shows that the cleaned dataset contains no remaining missing values in the main columns.

---

# 🚨 Part B — Outlier Treatment

A separate copy is created for outlier processing.

```python
DF_outliers = DF_imput.copy()
```

The notebook uses multiple approaches for handling outliers:

1. Z-Score
2. IQR
3. Percentile-based analysis
4. Winsorization

---

# 18. Cholesterol — Z-Score Method

Z-scores are calculated for cholesterol.

```python
z_scores = np.abs(
    stats.zscore(DF_outliers['cholesterol'])
)

DF_outliers = DF_outliers[z_scores < 3]
```

A threshold of:

```text
|Z-score| < 3
```

is used to remove extreme cholesterol observations.

---

# 19. Cholesterol Before vs After

The notebook compares cholesterol distributions before and after Z-score filtering using boxplots.

```python
cholesterol_before = DF['cholesterol'].copy()

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 6)
)

fig.suptitle(
    'Cholesterol Outliers: Before vs. After Z-Score Filter',
    fontsize=16,
    weight='bold'
)

sns.boxplot(
    y=cholesterol_before,
    ax=axes[0],
    color='lightcoral',
    flierprops={
        "marker": "o",
        "markerfacecolor": "red",
        "markersize": 6
    }
)

axes[0].set_title(
    'Cholesterol (Before Z-score < 3)',
    fontsize=14
)

axes[0].set_ylabel(
    'Cholesterol Level (mg/dL)',
    fontsize=12
)

sns.boxplot(
    y=DF_outliers['cholesterol'],
    ax=axes[1],
    color='lightgreen'
)

axes[1].set_title(
    'Cholesterol (After Z-score < 3)',
    fontsize=14
)

axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
```

---

# 20. BMI — IQR Method

The first quartile and third quartile are calculated.

```python
Q1 = DF_outliers['bmi'].quantile(0.25)

Q3 = DF_outliers['bmi'].quantile(0.75)
```

The Interquartile Range is calculated.

```python
IQR = Q3 - Q1
```

Lower and upper bounds are calculated using the standard IQR rule.

```python
lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR
```

Rows outside these boundaries are removed.

```python
DF_outliers = DF_outliers[
    (DF_outliers['bmi'] >= lower_bound) &
    (DF_outliers['bmi'] <= upper_bound)
]
```

---

# 21. BMI Before vs After

The notebook visualizes BMI before and after IQR-based filtering.

```python
bmi_before = DF['bmi'].copy()

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 6)
)

fig.suptitle(
    'BMI Outliers: Before vs. After IQR Method',
    fontsize=16,
    weight='bold'
)

sns.boxplot(
    y=bmi_before,
    ax=axes[0],
    color='lightcoral',
    flierprops={
        "marker": "o",
        "markerfacecolor": "red",
        "markersize": 6
    }
)

axes[0].set_title(
    'BMI (Original)',
    fontsize=14
)

axes[0].set_ylabel(
    'Body Mass Index (BMI)',
    fontsize=12
)

sns.boxplot(
    y=DF_outliers['bmi'],
    ax=axes[1],
    color='lightgreen'
)

axes[1].set_title(
    'BMI (After IQR Filter)',
    fontsize=14
)

axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
```

---

# 22. Blood Pressure — Percentile Method

The 1st and 99th percentiles are calculated.

```python
p1 = DF_outliers['blood_pressure'].quantile(0.01)

p99 = DF_outliers['blood_pressure'].quantile(0.99)
```

These values can be used as lower and upper limits for percentile capping.

The notebook also compares blood pressure before and after treatment using boxplots.

```python
blood_pressure_before = DF['blood_pressure'].copy()

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 6)
)

fig.suptitle(
    'Blood Pressure: Before vs. After Percentile Capping',
    fontsize=16,
    weight='bold'
)

sns.boxplot(
    y=blood_pressure_before,
    ax=axes[0],
    color='lightcoral',
    flierprops={
        "marker": "o",
        "markerfacecolor": "red",
        "markersize": 6
    }
)

axes[0].set_title(
    'Blood Pressure (Original)',
    fontsize=14
)

axes[0].set_ylabel(
    'Blood Pressure (mmHg)',
    fontsize=12
)

sns.boxplot(
    y=DF_outliers['blood_pressure'],
    ax=axes[1],
    color='lightgreen'
)

axes[1].set_title(
    'Blood Pressure (Capped at 1st & 99th %)',
    fontsize=14
)

axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
```

---

# 23. Winsorization

Winsorization is applied to:

```text
bmi
blood_pressure
cholesterol
glucose
```

The dataset is copied before applying Winsorization.

```python
DF_winsorized = DF_imput.copy()

cols_to_winsorize = [
    'bmi',
    'blood_pressure',
    'cholesterol',
    'glucose'
]
```

Winsorization is then applied using 5% limits on both sides.

```python
for col in cols_to_winsorize:
    DF_winsorized[col] = winsorize(
        DF_winsorized[col],
        limits=[0.05, 0.05]
    )
```

A message is printed to confirm the operation.

```python
print("Winsorization applied successfully to all columns.")

print(
    f"Dataset shape remains unchanged: "
    f"{DF_winsorized.shape}"
)
```

---

# 24. Winsorization Visualization

The notebook compares the original and Winsorized values using boxplots.

```python
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(20, 10)
)

fig.suptitle(
    'Winsorization Applied to All Features: Before vs. After',
    fontsize=18,
    weight='bold'
)

for i, col in enumerate(cols_to_winsorize):

    sns.boxplot(
        y=DF_imput[col],
        ax=axes[0, i],
        color='lightcoral',
        flierprops={
            "marker": "o",
            "markerfacecolor": "red",
            "markersize": 5
        }
    )

    axes[0, i].set_title(
        f'{col}\n(Original with Outliers)',
        fontsize=12
    )

    axes[0, i].set_ylabel('')

    sns.boxplot(
        y=DF_winsorized[col],
        ax=axes[1, i],
        color='lightskyblue'
    )

    axes[1, i].set_title(
        f'{col}\n(After Winsorization)',
        fontsize=12
    )

    axes[1, i].set_ylabel('')

plt.tight_layout(
    rect=[0, 0.03, 1, 0.92]
)

plt.show()
```

---

# 📦 Part C — Final Dataset

The final dataset is created from the outlier-treated dataset.

```python
DF_Final = DF_outliers.copy()
```

The final dataset can be displayed using:

```python
DF_Final
```

---

# 💾 25. Export Cleaned Dataset

The final cleaned dataset is exported to CSV.

```python
print(DF_Final.info())

DF_Final.to_csv(
    'output.csv',
    index=False
)
```

The output file is:

```text
output.csv
```

---

# 🧠 Data Cleaning Techniques Used

| Data Problem               | Column(s)                     | Technique                   |
| -------------------------- | ----------------------------- | --------------------------- |
| Missing numerical values   | BMI                           | Mean Imputation             |
| Missing categorical values | Region                        | Most Frequent               |
| Missing categorical values | Gender                        | Mode                        |
| Missing age                | Age                           | Random Sample Imputation    |
| Missing numerical values   | Cholesterol                   | KNN Imputation              |
| Missing numerical values   | Glucose                       | Iterative Imputation / MICE |
| Extreme cholesterol values | Cholesterol                   | Z-Score                     |
| Extreme BMI values         | BMI                           | IQR                         |
| Blood pressure extremes    | Blood Pressure                | Percentile approach         |
| Extreme numerical values   | BMI, BP, Cholesterol, Glucose | Winsorization               |

---

# 📌 Key Findings

## Effective Imputation Strategy

The project uses different imputation methods depending on the type of variable.

**Simple mean/mode imputation** is used for variables such as BMI, region, and gender.

**KNN and Iterative Imputation** are used for numerical health variables such as cholesterol and glucose.

The notebook identifies KNN and MICE/Iterative Imputer as effective approaches for these medical attributes because the numerical variables can contain relationships useful for estimating missing values.

---

## Best Outlier Handling Method

Several outlier-handling methods are explored:

* Z-Score
* IQR
* Percentile-based treatment
* Winsorization

The notebook notes that Winsorization and percentile capping have the advantage of preserving dataset size, while Z-score and IQR filtering can remove observations.

---

## Usability Improvement

After preprocessing:

* Missing values are removed
* Numerical variables are cleaned
* Extreme observations are handled
* The dataset becomes more suitable for machine-learning algorithms

The notebook notes that a cleaned dataset can help algorithms such as Logistic Regression and Random Forest work without errors caused by missing data.

---

# ▶️ How to Run the Project

## Step 1 — Install Python

Install Python 3.x on your computer.

Check the installation:

```bash
python --version
```

---

## Step 2 — Install Required Libraries

Run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

## Step 3 — Place the Dataset

Keep the dataset in the same folder as the notebook:

```text
patient_health_records.csv
```

---

## Step 4 — Open Jupyter Notebook

Run:

```bash
jupyter notebook
```

Then open:

```text
Data Cleanser(2).ipynb
```

---

## Step 5 — Run All Cells

Run the notebook from top to bottom.

The notebook will:

```text
Load Dataset
      ↓
Explore Dataset
      ↓
Check Missing Values
      ↓
Impute Missing Values
      ↓
Detect Outliers
      ↓
Handle Outliers
      ↓
Create Final Dataset
      ↓
Export output.csv
```

---

# 📤 Output

After successful execution, the cleaned dataset will be saved as:

```text
output.csv
```

The CSV file can then be used for:

* Data analysis
* Data visualization
* Machine learning
* Statistical analysis
* Predictive modeling

---

# 📁 Final Project Structure

```text
Data-Cleanser/
│
├── Data Cleanser(2).ipynb
│
├── patient_health_records.csv
│
├── output.csv
│
└── README.md
```

---

# 👨‍💻 Conclusion

The **Data Cleanser** project demonstrates a complete data preprocessing workflow for patient health records.

The project handles missing values using multiple imputation techniques, including:

* Mean Imputation
* Mode Imputation
* Random Sample Imputation
* KNN Imputation
* Iterative Imputation / MICE

It also demonstrates several outlier-handling techniques:

* Z-Score filtering
* IQR filtering
* Percentile-based treatment
* Winsorization

Finally, the processed dataset is exported as `output.csv`, making it ready for further analysis and machine-learning applications.

---

## ⭐ Project Workflow Summary

```text
Patient Health Dataset
        ↓
Data Loading
        ↓
Data Exploration
        ↓
Missing Value Detection
        ↓
Missing Value Imputation
        ↓
Outlier Detection
        ↓
Outlier Treatment
        ↓
Data Validation
        ↓
Final Clean Dataset
        ↓
output.csv
```

---

## 📜 License

This project is intended for educational and data-analysis purposes.

**Author:** Man Trivedi
