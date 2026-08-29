import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/laptopData.csv')

print(df.head())
print(df.shape)# 1303 X 12
print(df.columns)

target = 'Price'
df = df.drop('Unnamed: 0', axis=1)

# Missing values

miss = df.isnull().sum().sort_values(ascending=False).head(10)
percent = (df.isnull().sum() / len(df) * 100).round(2)
print(percent)
# dataga qaraganimda butunlay qator yoq edi
# i botdayam juda edailniy chiqdi

print(df[df.isnull().all(axis=1)])
df = df.dropna(how='all')


# ID ob tahsaldi va 30 pustoy row obtashaldi
print(df.shape) # 1273 X 11

miss = df.isnull().sum().sort_values(ascending=False).head(15)
print(miss) # pustoy qomadi deb oylagandim :(

for col in df.columns:
    print(df.dtypes[col]) # bitta price float, qogani str, casting str -> numeric



df = df.replace('?', np.nan)


df['Inches'] = pd.to_numeric(df['Inches'])

df['Ram'] = df['Ram'].str.replace('GB', '') # yonidan gb obtashadik
df['Ram'] = pd.to_numeric(df['Ram'])

df['Weight'] = df['Weight'].str.replace('kg', '')# yonidan kg obtashadik
df['Weight'] = pd.to_numeric(df['Weight'])

#Unable to parse string "?" at position 465, str dan numericga otqazoyatganimda ? belgi bor ekan nma qilay dedi
# print((df == '?').sum()) # bittadan soroq belgisi bor ekan

miss = df.isnull().sum().sort_values(ascending=False).head(15)
print(miss)

df['Inches'] = df['Inches'].fillna(df['Inches'].median())
df['Memory'] = df['Memory'].fillna(df['Memory'].mode()[0]) # chunki bu categorical
df['Weight'] = df['Weight'].fillna(df['Weight'].median())

miss = df.isnull().sum().sort_values(ascending=False).head(15)
print(miss) # endi tochna qomadi

# Duplicates
df = df.drop_duplicates(keep='first')
print(df.shape) # 1244 X 11

# Outliers
numeric_columns = ['Inches', 'Weight', 'Price'] # Ram dagi malumotlar outlier emas, huddi seatsga ohshab :)

for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    ]

print(df.shape) # 1126 X 11


for col in numeric_columns:
    print(df[col].describe())

print(df['Ram'].describe())


df.to_csv('data/clean_laptopData.csv', index=False)