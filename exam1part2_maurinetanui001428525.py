# Exam1Part2 - Maurine Tanui 001428525

# =============================================================================
# 1. Import Data from Part 1
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
path = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
df = pd.read_csv(path)

# Preview data
print(df.head())


# =============================================================================
# 2. Analyzing Individual Feature Patterns Using Visualization
# =============================================================================

# Check data types
print(df.dtypes)

# Q1: What is the data type of the column "peak-rpm"?
print("Data type of 'peak-rpm':", df['peak-rpm'].dtype)

# Correlation matrix (numeric columns)
correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
print(correlation_matrix)

# Q2: Correlation between specific variables
print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

# Visualizations - Scatterplots with regression lines
sns.regplot(x="engine-size", y="price", data=df)
plt.title("Engine Size vs Price")
plt.show()

print(df[["engine-size", "price"]].corr())

sns.regplot(x="highway-mpg", y="price", data=df)
plt.title("Highway MPG vs Price")
plt.show()

print(df[["highway-mpg", "price"]].corr())

sns.regplot(x="peak-rpm", y="price", data=df)
plt.title("Peak RPM vs Price")
plt.show()

print(df[['peak-rpm','price']].corr())

# Q3a: Stroke vs Price
print(df[["stroke", "price"]].corr())

# Q3b: Visualize stroke vs price
sns.regplot(x="stroke", y="price", data=df)
plt.title("Stroke vs Price")
plt.show()

# Categorical Variables - Boxplots
sns.boxplot(x="body-style", y="price", data=df)
plt.title("Body Style vs Price")
plt.show()

sns.boxplot(x="engine-location", y="price", data=df)
plt.title("Engine Location vs Price")
plt.show()

sns.boxplot(x="drive-wheels", y="price", data=df)
plt.title("Drive Wheels vs Price")
plt.show()


# =============================================================================
# 3. Descriptive Statistical Analysis
# =============================================================================

# Describe numerical features
print(df.describe())

# Describe object (categorical) features
print(df.describe(include=['object']))

# Value counts for drive-wheels
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

# Value counts for engine-location
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts.head(10))


# =============================================================================
# 4. Basics of Grouping
# =============================================================================

# Average price by drive-wheels
df_group_one = df[['drive-wheels', 'body-style', 'price']]
df_group_one['body-style'] = df_group_one['body-style'].map({
    'convertible': 1, 'hatchback': 2, 'sedan': 3, 'wagon': 4, 'hardtop': 5
})
df_grouped = df_group_one.groupby(['drive-wheels'], as_index=False).mean()
print(df_grouped)

# Group by drive-wheels and body-style
grouped_test1 = df[['drive-wheels', 'body-style', 'price']].groupby(
    ['drive-wheels', 'body-style'], as_index=False).mean()

# Pivot table
grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')
grouped_pivot = grouped_pivot.fillna(0)
print(grouped_pivot)

# Q4: Group by body-style
avg_price_body_style = df[['body-style', 'price']].groupby(['body-style'], as_index=False).mean()
print(avg_price_body_style)

# Heatmap
plt.pcolor(grouped_pivot, cmap='Pastel2_r')
plt.colorbar()
plt.title("Heatmap: Drive Wheels and Body Style vs Price")
plt.show()

# Custom labels
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='Pastel2_r')

ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5)

ax.set_xticklabels(grouped_pivot.columns.levels[1], rotation=90)
ax.set_yticklabels(grouped_pivot.index)

fig.colorbar(im)
plt.title("Heatmap with Labels")
plt.show()


# =============================================================================
# 5. Correlation and Causation
# =============================================================================

# Pearson Correlation + P-values
def print_corr(var):
    pearson_coef, p_value = stats.pearsonr(df[var], df['price'])
    print(f"{var} vs price --> Pearson Coefficient: {pearson_coef:.3f}, P-value: {p_value:.3e}")

variables_to_test = [
    'wheel-base', 'horsepower', 'length', 'width', 'curb-weight',
    'engine-size', 'bore', 'city-mpg', 'highway-mpg'
]

for var in variables_to_test:
    print_corr(var)

# Final thoughts:
important_continuous = [
    'length', 'width', 'curb-weight', 'engine-size', 'horsepower',
    'city-mpg', 'highway-mpg', 'wheel-base', 'bore'
]
important_categorical = ['drive-wheels']
