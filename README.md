# House Price Prediction

Internship Project — Week 1

A regression analysis that predicts house prices from property features (size, rooms, amenities, furnishing status) and identifies which features influence price the most.

## Problem Statement

Real estate buyers and sellers often rely on guesswork or outdated comparisons to estimate a property's fair value. This project builds a regression model that predicts house prices based on property features, then identifies which features most strongly influence price.

## Dataset

- **Source:** [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) (Kaggle)
- **File:** `Housing.csv`
- **Size:** 545 rows × 13 columns
- **Target column:** `price`
- **Feature columns:**
  - Numeric: `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
  - Binary (yes/no): `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`
  - Categorical: `furnishingstatus` (furnished / semi-furnished / unfurnished)
- **Data quality:** No missing values, no duplicate rows.

## Project Structure

```
HousePricePrediction_[YourName]/
├── analysis.ipynb          # Full notebook: all 5 tasks, executed with outputs
├── Housing.csv             # Dataset used
├── summary.docx            # 1-page written summary of findings
├── charts/
│   ├── chart1_price_distribution.png
│   ├── chart2_correlation_heatmap.png
│   ├── chart3_actual_vs_predicted.png
│   └── bonus_feature_importance.png
└── README.md
```

## Tasks Completed

### Task 1 — Data Loading & Exploration
Loaded the CSV with Pandas, inspected the first 10 rows, confirmed shape (545 rows × 13 columns), identified `price` as the target and the remaining 12 columns as features, and checked for missing values (none found).

### Task 2 — Data Cleaning
- Checked and handled missing values (none present; median/mode-fill logic included defensively)
- Removed duplicate rows (none found)
- Converted binary yes/no columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) to 0/1
- One-hot encoded `furnishingstatus` (drop-first to avoid multicollinearity)
- Retained all columns as meaningful predictors

### Task 3 — Model Building
Split the data 80/20 (train/test, `random_state=42`) and trained two models:

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 970,043 | 1,324,507 | **0.653** |
| Random Forest Regressor | 1,022,560 | 1,401,497 | 0.611 |

Linear Regression slightly outperformed Random Forest — a sign that price scales fairly linearly with the available features at this sample size (545 rows is small for Random Forest to exploit non-linear splits effectively).

### Task 4 — Visualization
Three required charts plus one bonus chart, all saved to `charts/`:

1. **Price distribution histogram** — right-skewed; most homes cluster between 3M–6M, with a long tail up to 13.3M.
2. **Correlation heatmap** — `area` (0.54), `bathrooms` (0.52), `airconditioning` (0.45), and `stories` (0.42) correlate most strongly with price.
3. **Actual vs. predicted scatter plot** (Linear Regression) — predictions track closely with actuals in the mid-range, but the model under-predicts the priciest homes.
4. *(Bonus)* Random Forest feature importance bar chart — `area` (~47%) and `bathrooms` (~15%) dominate.

### Task 5 — Insights & Summary

**Which features influence house price the most?**
`area` is the single strongest driver (highest correlation and highest Random Forest importance), followed by `bathrooms` and `airconditioning`.

**How accurate was the model?**
The Linear Regression model explains about 65% of price variation (R² ≈ 0.65), with predictions typically within ~970,000 of the actual price — useful for a quick estimate, not a substitute for a professional appraisal.

**What was surprising?**
Random Forest, despite being more flexible, performed slightly *worse* than plain Linear Regression — suggesting the price relationship in this dataset is mostly linear rather than driven by complex feature interactions.

**Recommendation for a real estate business:**
Prioritize accurate area measurements and bathroom counts in every listing, since errors there skew valuations the most. Use a simple linear model as a first-pass pricing tool, but supplement it with local comparables for high-end properties, where model accuracy drops off noticeably.

## Tools & Libraries

| Tool | Purpose |
|---|---|
| Python 3.x | Main programming language |
| Jupyter Notebook | Writing and running the analysis |
| Pandas / NumPy | Data loading and cleaning |
| Scikit-learn | Regression models and evaluation (`LinearRegression`, `RandomForestRegressor`, `train_test_split`, `mean_absolute_error`, `mean_squared_error`, `r2_score`) |
| Matplotlib / Seaborn | Charts and visualization |

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook analysis.ipynb
```

Run all cells top to bottom. Generated charts are saved automatically to the `charts/` folder.

## License

For educational/internship use.
