import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

st.title("Automobile Data Analysis - Exam Part 2 📊")

# =============================================================================
# 1. Load Data
# =============================================================================
st.header("1. Load and Preview Data")

DATA_URL = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
df = pd.read_csv(DATA_URL)

st.write("### Preview of Dataset")
st.dataframe(df.head())

# =============================================================================
# 2. Individual Feature Analysis
# =============================================================================
st.header("2. Feature Patterns & Correlation Analysis")

st.subheader("Data Types")
st.write(df.dtypes)

st.write("### Data Type of 'peak-rpm'")
st.write(df['peak-rpm'].dtype)

st.write("### Numeric Correlation Matrix")
correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
st.dataframe(correlation_matrix)

st.write("### Selected Features Correlation")
st.dataframe(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

# Plots
st.subheader("Scatter Plots with Regression Lines")

def scatter_plot(x, y, title):
    fig, ax = plt.subplots()
    sns.regplot(x=x, y=y, data=df, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)
    st.write(df[[x, y]].corr())

scatter_plot("engine-size", "price", "Engine Size vs Price")
scatter_plot("highway-mpg", "price", "Highway MPG vs Price")
scatter_plot("peak-rpm", "price", "Peak RPM vs Price")
scatter_plot("stroke", "price", "Stroke vs Price")

# Categorical Variables - Boxplots
st.subheader("Boxplots for Categorical Features")

def box_plot(x, y):
    fig, ax = plt.subplots()
    sns.boxplot(x=x, y=y, data=df, ax=ax)
    ax.set_title(f"{x.title()} vs {y.title()}")
    st.pyplot(fig)

box_plot("body-style", "price")
box_plot("engine-location", "price")
box_plot("drive-wheels", "price")

# =============================================================================
# 3. Descriptive Statistical Analysis
# =============================================================================
st.header("3. Descriptive Statistics")

st.subheader("Numeric Columns")
st.dataframe(df.describe())

st.subheader("Categorical Columns")
st.dataframe(df.describe(include=['object']))

st.subheader("Drive-Wheels Value Counts")
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
st.dataframe(drive_wheels_counts)

st.subheader("Engine Location Value Counts")
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
st.dataframe(engine_loc_counts)

# =============================================================================
# 4. Grouping and Pivot Tables
# =============================================================================
st.header("4. Grouping & Pivot Tables")

df_group_one = df[['drive-wheels', 'body-style', 'price']].copy()
df_group_one['body-style'] = df_group_one['body-style'].map({
    'convertible': 1, 'hatchback': 2, 'sedan': 3, 'wagon': 4, 'hardtop': 5
})
df_grouped = df_group_one.groupby(['drive-wheels'], as_index=False).mean()
st.write("### Average Price by Drive-Wheels")
st.dataframe(df_grouped)

grouped_test1 = df[['drive-wheels', 'body-style', 'price']].groupby(
    ['drive-wheels', 'body-style'], as_index=False).mean()
grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style').fillna(0)
st.write("### Pivot Table")
st.dataframe(grouped_pivot)

# Heatmap
st.subheader("Heatmap of Drive Wheels & Body Style vs Price")

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='Pastel2_r')
fig.colorbar(im)

ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5)

ax.set_xticklabels(grouped_pivot.columns.levels[1], rotation=90)
ax.set_yticklabels(grouped_pivot.index)

ax.set_title("Heatmap with Labels")
st.pyplot(fig)

# =============================================================================
# 5. Correlation and Causation
# =============================================================================
st.header("5. Correlation and Causation")

def show_correlation(var):
    pearson_coef, p_value = stats.pearsonr(df[var], df['price'])
    st.write(f"**{var} vs price**: Pearson Coefficient = {pearson_coef:.3f}, P-value = {p_value:.3e}")

variables_to_test = [
    'wheel-base', 'horsepower', 'length', 'width', 'curb-weight',
    'engine-size', 'bore', 'city-mpg', 'highway-mpg'
]

for var in variables_to_test:
    show_correlation(var)

st.subheader("Important Features for Prediction")
st.markdown("""
**Continuous:**  
`length`, `width`, `curb-weight`, `engine-size`, `horsepower`, `city-mpg`, `highway-mpg`, `wheel-base`, `bore`

**Categorical:**  
`drive-wheels`
""")
