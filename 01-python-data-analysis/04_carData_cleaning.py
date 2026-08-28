import opendatasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cardekho.csv')

task1 = 'task1'
print(df.isnull().sum().sort_values(ascending=False).head(10))
print(df.shape)
#(8128, 12)



df = df.drop_duplicates()
print(df.isnull().sum().sort_values(ascending=False).head(10))
print(df.shape) #  no duplicates

miss = (df.isnull().sum() / len(df) * 100).round(2)
x = miss[miss > 0].sort_values(ascending=False).head(10)
print(x)# missing procenti topildi


#filling proces nachalsya
print(df['max_power'].dtypes) # problem only with max_power because its str
print(df['mileage(km/ltr/kg)'].dtypes)
print(df['engine'].dtypes)
print(df['seats'].dtypes)


df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce') #converts str to numeric
a = ['max_power', 'mileage(km/ltr/kg)', 'engine', 'seats']

for col in a:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

print(df[a].isnull().sum()) # ne ostalos pustih rows


# teper remove outliers

for col in a:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    before = df.shape[0]
    df = df[
        (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    ]
    after = df.shape[0]

    print(col, 'removed: ', before - after )

print(df.shape)


task2 = 'task2'
sns.boxplot(
    data=df,
    x='fuel',
    y='selling_price',
)
plt.show()
#fuel type selling pricega tasiri


sns.histplot(
    data=df,
    x='fuel',
)
plt.show()
# nechta moshina shu yoqilgida yurishi



task3 = 'task3'
print(df.select_dtypes(include='number').columns)

numeric_columns = ['year', 'selling_price', 'km_driven', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']
#print dan output oldim

for col in numeric_columns:
    print(df[col].describe())



# count    5.186000e+03
# mean     6.889846e+04
# std      5.372434e+04
# min      1.000000e+00
# 25%      3.500000e+04
# 50%      6.000000e+04
# 75%      9.267725e+04
# max      2.360457e+06
# Name: km_driven, dtype: float64

km_min = 1
km_max = 2360557
# bitta moshina 1 km yurgan, bitta moshina 2 million km kop yurgan


# count    5.186000e+03
# mean     4.107080e+05
# std      2.477146e+05
# min      3.000000e+04
# 25%      2.200000e+05
# 50%      3.550000e+05
# 75%      5.500000e+05
# max      2.150000e+06
# Name: selling_price, dtype: float64

mean = 4.107080
median = 3.55000
# very expensive car pulling mean



# count    5186.0
# mean        5.0
# std         0.0
# min         5.0
# 25%         5.0
# 50%         5.0
# 75%         5.0
# max         5.0
# Name: seats, dtype: float64

# bunisida manimcha outlier clean qilayotganimda boshqacha seatla filtrdan otomagan