# Automotives — Vehicle Pricing Analysis

An interactive Streamlit app that analyzes automobile pricing data to identify which features most influence vehicle price, combining exploratory data analysis, statistical testing, and predictive modeling.

# What it does

- Explores relationships between vehicle features (engine size, horsepower, mileage, brand, etc.) and price
- Runs statistical tests to validate which features are significantly associated with price
- Surfaces findings through an interactive Streamlit interface


## Tech stack
Python, pandas, scikit-learn (if used), Streamlit, [stats library — e.g. scipy/statsmodels]

## How to run

```bash
git clone https://github.com/MaurineTanui/Automotives.git
cd Automotives
pip install -r requirements.txt
streamlit run app.py
```




Key contributions:

Performed exploratory data analysis to evaluate how vehicle attributes influence price

Visualized price differences across product categories using boxplots and regression analysis

Built correlation tables and summary statistics to support pricing insights

Used grouping and pivot tables to compare average prices across drive types and body styles

Highlighted key features that pricing teams can use for competitive positioning and market analysis
