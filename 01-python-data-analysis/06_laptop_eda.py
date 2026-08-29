import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/clean_laptopData.csv')

# 1 Load
print(df.columns)
print(df.info())
print(df.describe())

print(df.dtypes.value_counts())

# 2 missing || 3 histplot, kdeplot, boxplot, log1p

print(df.isnull().sum()) # yoq missing values

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))

sns.histplot(
    df['Price'],
    kde=True,
    color = 'blue',
    ax = axes[0]
)
axes[0].set_title('KDE skew')

sns.histplot(
    df['Price'],
    fill=True,
    ax = axes[1]
)
axes[1].set_title('KDE chizigi')

sns.boxplot(
    y=df['Price'],
    ax = axes[2],
)
axes[2].set_title('Boxplot')

plt.tight_layout()
plt.show()


# price juda katta loq1p ishlatish kere (Transformatsiya)
df['Price_log'] = np.log1p(df['Price'])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

sns.histplot(
    df['Price'],
    kde=True,
    color = 'blue',
    ax = axes[0]
)
axes[0].set_title('Before Log')

sns.histplot(
    df['Price_log'],
    kde=True,
    ax = axes[1],
    color= 'red',
)
axes[1].set_title('After Log')

plt.tight_layout()
plt.show()


# 4 Correlation
correlation = df.select_dtypes(include='number').corr()

top10 = correlation['Price'].drop(columns='Price').abs().sort_values(ascending=False).head(10).index

top_cols = list(top10) + ['Price']

#heatmap
mask = np.triu(
    np.ones_like(correlation[top_cols].loc[top_cols], dtype=bool) # mask symmetric joyini korsatmas ekan
)

sns.heatmap(
    correlation.loc[top_cols, top_cols],
    annot=True,
    mask=mask,
    center=0,
    vmin=-1,
    vmax=1,
)
plt.title('Top 10 correlations')
plt.tight_layout()
plt.show()


# 5 categorical + barplot
categorical_columns = ['Company', 'TypeName', 'OpSys']

for col in categorical_columns:
    # Countplot
    plt.figure(figsize=(10, 5))
    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index
    )

    plt.xticks(rotation=45)
    plt.title(f'Number of Laptops by {col}')
    plt.tight_layout()
    plt.show()



    # Average Price barplot
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df,
        x=col,
        y='Price',
        estimator='mean'
    )


    plt.xticks(rotation=45)
    plt.title(f'Average Price by {col}')
    plt.tight_layout()
    plt.show()


# 6 scatter + reg
numeric_features = ['Ram', 'Weight', 'Inches']

for col in numeric_features:
    plt.figure(figsize=(7, 5))
    sns.regplot(
        data=df,
        x=col,
        y='Price',
        scatter_kws={'alpha': 0.7},
        line_kws={'color': 'red'}
    )


    plt.title(f'{col} vs Price')
    plt.tight_layout()
    plt.show()

for col in numeric_features:
    r = df[col].corr(df['Price'])
    print(f'{col} vs Price: r = {r}')