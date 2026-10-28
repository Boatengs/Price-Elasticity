# Price Elasticity and Promotion Analysis Dashboard

A Streamlit dashboard built for my capstone in the MPS in Applied Machine Intelligence at the Roux Institute, Northeastern University. It looks at how a set of brands price their products, how deep and how often they promote, and how unit sales respond to price.

This was a team project (Capstone Team C). Brands appear as A through E in the app.

## What the dashboard does

The app reads a promotional sales dataset (CSV), lets you filter by brand and subcategory, and splits the analysis across six tabs.

### Summary

The team's written findings on competitive reactions, market leaders, promotion effectiveness, loyalty, price sensitivity, seasonality and premium versus budget positioning, followed by recommendations and a glossary of the column names.

### Overview

Total units by brand, and average promotional price by week.

### Promotions

Average discount depth by brand, the ten lowest selling subcategories, the share of each brand's units sold on promotion, and promoted units by brand over time.

### Modeling

Predicts revenue (`$`) from average unit price, base price, promotional price, discount percentage and `tdp`. You can switch between linear regression, ridge regression, random forest and XGBoost. Each model trains on an 80/20 split and reports R² and RMSE with an actual versus predicted plot.

### Advanced Metrics

K-means segmentation of products into three groups on price and `tdp`, a price sensitivity curve (mean units sold across five price quantiles), and monthly averages for price, promotional price and discount when a `date` column is present.

### Download

Export the filtered data as CSV.

## Data

The dataset isn't included in this repository. The app asks you to upload it when it starts. It expects these columns:

| Column | Meaning |
|---|---|
| `brand`, `sub_category` | Filters and grouping |
| `units`, `any_promo_units` | Units sold, total and on promotion |
| `$` | Revenue |
| `avg_unit_price`, `base_unit_price`, `any_promo_unit_price` | Average, regular and promotional unit price |
| `any_promo_unit_price_%_disc` | Promotional discount percentage |
| `tdp` | Defined in the app's glossary as Total Display Price |
| `week`, `date` | Time fields for trend charts |

Each chart is skipped if the columns it needs are missing, so the app still runs on a partial file.

## Run locally

```bash
git clone https://github.com/Boatengs/Price-Elasticity.git
cd Price-Elasticity
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python, Streamlit, pandas, Plotly, scikit-learn, XGBoost

## Limitations

* The price sensitivity view compares mean units across price bins. It shows the direction of the relationship but isn't a formal elasticity estimate with controls for promotion, seasonality or distribution.
* The revenue models use a single random train/test split rather than a time based split, so the reported scores may be optimistic for forecasting future weeks.
