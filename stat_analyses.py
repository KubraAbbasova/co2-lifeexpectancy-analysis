# stat_analysis.py
import pandas as pd

file_path = "/Users/kubra/Desktop/ma705-final project/final_health_env.csv"
df = pd.read_csv(file_path)
print(df.head())



print("Shape of dataset:", df.shape)
print(" First 5 rows")
print(df.head())

# Data overview
print(df.dtypes)
print(df.isnull().sum())

# -----------------------------------------
#  Descriptive Statistics
print("DESCRIPTIVE STATISTICS")

print(df['life_expectancy'].describe())

print(df['emissions'].describe())

# Grouped by Year (optional summary)
print("Grouped Life Expectancy by Year (Mean):")
print(df.groupby('year')['life_expectancy'].mean().round(2))

print("Grouped Emissions by Year (Mean):")
print(df.groupby('year')['emissions'].mean().round(2))

#Bar Plot of Average CO₂ Emissions per Year
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind

avg_emissions_per_year = df.groupby("year")["emissions"].mean().reset_index()

# Create the bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=avg_emissions_per_year, x="year", y="emissions", palette="Blues_d")
plt.title("Average CO₂ Emissions per Year (2000–2021)")
plt.xlabel("Year")
plt.ylabel("Average Emissions (MtCO₂e)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("barplot_avg_emissions_per_year.png")
plt.show()


# -----------------------------------------
# Plot Life Expectancy Over Time
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='year', y='life_expectancy', estimator='mean')
plt.title("🌍 Average Life Expectancy Over Time")
plt.xlabel("Year")
plt.ylabel("Life Expectancy (Years)")
plt.grid(True)
plt.tight_layout()
plt.savefig("life_expectancy_over_time.png")
plt.show()
plt.figure(figsize=(8, 5))
sns.histplot(df['life_expectancy'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Life Expectancy")
plt.xlabel("Life Expectancy (Years)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("hist_life_expectancy.png")
plt.show()


# -----------------------------------------
# Plot 2: Emissions vs Life Expectancy
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='emissions', y='life_expectancy', hue='year', palette='viridis', alpha=0.7)
plt.title("💨 CO₂ Emissions vs Life Expectancy")
plt.xlabel("Emissions (MtCO₂e)")
plt.ylabel("Life Expectancy")
plt.grid(True)
plt.tight_layout()
plt.savefig("emissions_vs_life_expectancy.png")
plt.show()

# -----------------------------------------
# Correlation
print("Correlation Matrix")
print(df[['emissions', 'life_expectancy']].corr())



# -----------------------------------------
# Linear Regression
X = df[['emissions']]
X = sm.add_constant(X)  # Add intercept
y = df['life_expectancy']

model = sm.OLS(y, X).fit()
print(" Regression Results")
print(model.summary())

# -----------------------------------------
# T-test: Compare life expectancy in 2000 vs 2020
life_2000 = df[df['year'] == 2000]['life_expectancy']
life_2020 = df[df['year'] == 2020]['life_expectancy']

t_stat, p_val = ttest_ind(life_2020, life_2000, equal_var=False)
print(f" T-Test Comparing Life Expectancy (2000 vs 2020):")
print(f"T-statistic: {t_stat:.2f}, P-value: {p_val:.4f}")


from scipy.stats import f_oneway

# -----------------------------------------
#  Create Emission Groups for ANOVA
df['emission_group'] = pd.qcut(df['emissions'], q=3, labels=['Low', 'Medium', 'High'])

# Check group sizes (optional)
print("\n🔍 Emission Group Sizes:")
print(df['emission_group'].value_counts())

# -----------------------------------------
# One-Way ANOVA: Life Expectancy Across Emission Groups
low = df[df['emission_group'] == 'Low']['life_expectancy']
med = df[df['emission_group'] == 'Medium']['life_expectancy']
high = df[df['emission_group'] == 'High']['life_expectancy']

f_stat, p_val = f_oneway(low, med, high)
print(f" ANOVA Results:\nF-statistic = {f_stat:.2f}, p-value = {p_val:.4f}")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='emission_group', y='life_expectancy', palette='Set2')
plt.title("Life Expectancy by Emission Group")
plt.xlabel("Emission Group")
plt.ylabel("Life Expectancy")
plt.grid(True)
plt.tight_layout()
plt.savefig("life_expectancy_by_emission_group.png")
plt.show()

#advansed stat multilinear
# Example: Add 'year' as an additional predictor
X = sm.add_constant(df[['emissions', 'year']])
model_multi = sm.OLS(y, X).fit()
print(model_multi.summary())
