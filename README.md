# Automotives — Vehicle Pricing Analysis

An interactive Streamlit app that analyzes automobile pricing data to identify which features most influence vehicle price, combining exploratory data analysis, correlation testing, and visualization.

## Key Findings

Pearson correlation of each feature against price (201 vehicles, 29 features):

| Feature | Correlation (r) | Direction |
|---|---|---|
| **Engine size** | 0.872 | Strongest positive predictor of price |
| **Curb weight** | 0.834 | Heavier vehicles command higher prices |
| **Horsepower** | 0.810 | Strong positive relationship |
| **Width** | 0.751 | Larger vehicles trend more expensive |
| **Highway MPG** | -0.705 | More fuel-efficient vehicles trend cheaper |
| **Length** | 0.691 | Moderate positive relationship |
| **City MPG** | -0.687 | Same inverse pattern as highway MPG |
| **Wheelbase** | 0.585 | Moderate positive relationship |
| **Bore** | 0.543 | Weaker but still significant |

All correlations are statistically significant (p < 0.001).

**Takeaway:** engine size, weight, and horsepower *not fuel economy or wheelbase * are the dominant price drivers in this dataset, together explaining most of the price variation. Pricing/positioning strategy should weight these three features most heavily.

## What it does

- Explores relationships between vehicle features (engine size, horsepower, mileage, body style, drive-wheels, etc.) and price via scatter plots with regression lines and boxplots
- Runs Pearson correlation tests to quantify which features are significantly associated with price
- Builds grouped/pivot tables and a heatmap comparing average price across drive-wheel type and body style
- Surfaces all findings through an interactive Streamlit interface

## Tech stack

Python, pandas, NumPy, seaborn, matplotlib, scipy (Pearson correlation testing), Streamlit

## How to run

```
git clone https://github.com/MaurineTanui/Automotives.git
cd Automotives
pip install -r requirements.txt
streamlit run app.py
```

## Author

**Maurine Tanui** — MSc Analytics, Saint Louis University
[LinkedIn](https://www.linkedin.com/in/maurine-cherono-cpa-016a29196/) · [Portfolio](https://maurinetanui.github.io/)
alysis
