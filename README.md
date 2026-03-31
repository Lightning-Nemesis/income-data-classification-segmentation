# income-data-classification-and-segmentation

A machine learning project built on the 1994–1995 US Census Bureau Current Population Survey data. The project addresses two objectives:

- **Income Classifier** — Train and validate a binary classifier to predict whether an individual earns above or below $50,000 using 40 demographic and employment variables
- **Segmentation Model** — Build an unsupervised customer segmentation model and demonstrate how resulting groups differ for retail marketing purposes

---

## Project Structure

```
income-classification-and-segmentation/
│
├── data/
│   ├── censusbureau.data              # Raw comma-delimited data file
│   └── census-bureau.columns         # Column names header file
│  
├── model/
│   ├── catboost_model
│                                  # model -> catboost_model
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Classification.ipynb
│   └── 03_Segmentation.ipynb
│
├── requirements.txt
├── README.md
└── Report.pdf
```

---

## Environment

- Python 3.8 or higher
- Jupyter Notebook or Google Colab
- These notebooks were originally developed on Google Colab

---

## Installation

Clone the repository and install all required packages:

```bash
git clone https://github.com/Lightning-Nemesis/income-classification-and-segmentation.git
cd income-classification-and-segmentation
pip install -r requirements.txt
jupyter notebook
```

---

## Running the Notebooks

Launch Jupyter from the project root:

```bash
jupyter notebook
```

Run the notebooks in the following order:

---

### 1. EDA — `01_EDA.ipynb`

Exploratory data analysis of the census dataset before modeling.

**What it covers:**

1. Loading the raw data using the columns header file; identifying continuous columns (`age`, `wage per hour`, `capital gains`, `capital losses`, `dividends from stocks`, `num persons worked for employer`, `weeks worked in year`) and categorical columns
2. Handling missing values — replacing `NaN` and `?` placeholders with `Unknown`
3. **Univariate analysis**
   - Target variable: weighted income label distribution
   - Categorical variables: weighted bar plots for each feature
   - Continuous variables: weighted distribution plots and boxplots
4. **Bivariate analysis**
   - Categorical variables vs. income: weighted count plots split by income class, percentage of high earners by category
   - Continuous variables vs. income: weighted violin plots across income groups
   - Geographic analysis: choropleth map of weighted population counts by U.S. state of previous residence
5. **Multivariate analysis** — weighted Pearson correlation matrix across all continuous features and the income label

---

### 2. Classification — `02_Classification.ipynb`

Binary income classification pipeline from preprocessing through evaluation.

**What it covers:**

1. **Data preprocessing**
   - Data cleaning: target encoding, duplicate consolidation with weight summing, missing value treatment
   - Feature engineering: education ordinal encoding, categorical dtype assignment for CatBoost compatibility
   - Train/test split (80/20, stratified on income label)
   - Evaluation metric selection: weighted PR-AUC
2. **Modeling — baseline comparisons**
   - Logistic Regression (baseline, with PCA, with Weight of Evidence encoding)
   - Coefficient analysis
   - Random Forest
   - CatBoost
   - XGBoost
   - LightGBM
3. **CatBoost — winner**
   - Early stopping with eval set
   - Feature selection based on feature importances (24 features covering 95% of importance)
   - Hyperparameter tuning with Hyperopt (Bayesian TPE search, 30 trials)
   - Cross-validation stability check
   - Evaluation at tuned threshold (0.3377)
4. **Fairness analysis** — group-level PR-AUC, false negative rate, and false positive rate across sex and race
5. **Explainable AI (SHAP)** — feature contribution analysis on the final model

---

### 3. Segmentation — `03_Segmentation.ipynb`

Unsupervised customer segmentation built on top of the classification pipeline.

**What it covers:**

1. Running the trained CatBoost model on the full dataset to generate an `income_prediction_score` per record
2. **Feature engineering for clustering**
   - Continuous/numeric column transformations: skewed features converted to binary flags
   - Categorical column consolidation and encoding
   - VIF analysis to check multicollinearity and select features
3. **Segmentation model**
   - PCA — standardization and dimensionality reduction to 23 components (95% variance)
   - K-Means elbow method and silhouette scores for k = 2 through 10
   - GMM experiment (evaluated as alternative; K-Means selected based on silhouette performance)
4. **Final model: K-Means with k = 5** — cluster assignment and validation
5. **Customer profiling** — segment characterization by age, employment, education, filing status, income score, and marketing implications

---

## Results

**Classifier (CatBoost, threshold = 0.3377)**

| Metric | Score |
|---|---|
| PR-AUC | 0.7074 |
| ROC-AUC | 0.9569 |
| F1 | 0.65 |
| Precision | 0.68 |
| Recall | 0.62 |

**Segments (K-Means, k=5)**

| Segment | Label | Size |
|---|---|---|
| 0 | Dependents / Children | 28% |
| 1 | Core Working Families | 22% |
| 2 | Independent Workers | 20% |
| 3 | Older Non-Working Adults | 20% |
| 4 | High Income Group | 9% |
