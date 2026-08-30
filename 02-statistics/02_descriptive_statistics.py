import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("penguins").dropna()


# 1 mean/median/std/var/ taqqoslash kere describega

col = df.select_dtypes(include='number')
#bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
print(col)

# alohida
for x in col:
    print(x)
    print(df[x].mean())
    print(df[x].median())
    print(df[x].std())
    print(df[x].var())

# bill_length_mm
# 43.99279279279279
# 44.5
# 5.468668342647559
# 29.906333441875603

# bill_depth_mm
# 17.164864864864864
# 17.3
# 1.9692354633199007
# 3.877888309996744

# flipper_length_mm
# 200.96696696696696
# 197.0
# 14.015765288287879
# 196.44167661637542

# body_mass_g
# 4207.057057057057
# 4050.0
# 805.2158019428965
# 648372.487698542


print(col.describe())

#  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
# count      333.000000     333.000000         333.000000   333.000000
# mean        43.992793      17.164865         200.966967  4207.057057
# std          5.468668       1.969235          14.015765   805.215802
# min         32.100000      13.100000         172.000000  2700.000000
# 25%         39.500000      15.600000         190.000000  3550.000000
# 50%         44.500000      17.300000         197.000000  4050.000000
# 75%         48.600000      18.700000         213.000000  4775.000000
# max         59.600000      21.500000         231.000000  6300.000000


conclusion = 'ozimiz yozganda nuqtadan keyin koproq raqam yani more accurate'


# 2 Vizualitsiya

x = df.groupby('species')['bill_length_mm'].mean()
# species
# Adelie       38.823973
# Chinstrap    48.833824
# Gentoo       47.568067
# shunaqa qilib qaytarib berdi, this is mean

y = df.groupby('species')['bill_length_mm'].std()
# species
# Adelie       2.662597
# Chinstrap    3.339256
# Gentoo       3.106116
# this is std


print(x)
print(y)

plt.figure(figsize=(10,6))
plt.bar(x.index, x.values)


plt.errorbar(
    x.index,
    x.values,
    yerr=y.values,
    fmt='none',
    capsize=50
)
plt.xlabel('Species ')
plt.ylabel('Bill_length_mm mean')
plt.show()


# 3 z-score blan sort qilish (|z| > 2)  kere alohida df ga


# z = (x - mu) / sigma
# x = value from column
# mu =  mean
# sigma = std

mu = df['body_mass_g'].mean()
sigma = df['body_mass_g'].std()
lst = []

for x in df['body_mass_g']:
    z = (x - mu) / sigma

    if abs(z) > 2 :
        lst.append(z)


outliers_df = df.iloc[lst]
print(outliers_df)
# karoche bu yerda , man shu valuelani listga store qigandimu, osha valuedagi rowlani endi yangi dataframe ga store qildim