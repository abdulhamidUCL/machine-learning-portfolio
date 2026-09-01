import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")

# 1 Yuklash va
df = pd.read_csv('data/insurance.csv')

print(df.head())
print(df.shape) #(1338, 7)
print('================================================================================')



# 2 missing , filling, outliers

print(df.info())# hamma ustun typelari zor, casting kerakmas

missing = df.isnull().sum().sort_values(ascending=False)
print(missing) # bosh column la yoqlar
# shunga filling ham shart emas



print(df['charges'].describe()) # mean blan medianani farqi judaaa katta, outlierlaga tozalash kerak
# print(df['bmi'].describe()) # mean blan medianasi relatively similar va outlier yoq
print('================================================================================')

# outliers filter
Q1 = df['charges'].quantile(0.25)
Q3 = df['charges'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df['charges'] >= lower) & (df['charges'] <= upper)
]

print(df.shape) #(1199, 7)
print(df['charges'].describe()) # mean blan medianani farqi kichraydi
print('================================================================================')



# 3 vizual tahlil - Hist/Scatter/Boxplot/Heatmap

# 1 Histplot

plt.figure(figsize=(10, 6))
sns.histplot(df['charges'], bins=50, color='green', kde=True)
plt.xlabel('Charges')
plt.title('Charges Distribution')
plt.show()
# Xulosa: right-skewed yani ong tomonda uzun dum bop qogan, va malumotlar kopi chap tarafda yegilib qogan


df['charges_log'] = np.log1p(df['charges'])
plt.figure(figsize=(10, 6))
sns.histplot(df['charges_log'], bins=50, color='green', kde=True)
plt.xlabel('charges_log')
plt.title('charges_log Distribution')
plt.show()
# Log transform ishlatilgandan keying ancha normallashdi, (huge difference)



plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    y=df['charges_log'],
    x=df['smoker'],
)
plt.xlabel('smoker')
plt.ylabel('charges_log')
plt.title('Chakish holati tolovlarga bogliqligi')
plt.show()
# Xulosa: smokers have higher charges va medianasi ham balandroq,
# Lekin: hali statistika(hypothesis tesing) blan tasdiqlash kere buni :))



# heatmap
numeric = df.select_dtypes(include='number')

corr = numeric.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(
    corr,
    annot=True,
    annot_kws={'size': 10},
    cmap='RdYlGn',
)
plt.title('Correlation Matrix')
plt.show()
# Xulosa: yosh charges_log blan correlation bor ekan r ~ 0.63


# Statistik tahlil (hypothesis testing)

stat, p = stats.shapiro(
    df['charges_log'].sample(500, random_state=1)
)
print(f'p = {p:.9f}')# p = 0.000000089

if p < 0.05:
    print('Normal emas')
else:
    print('Normal')

# normal emas chiqdi


# Savol: Chekish holati individual tibbiy sug'urta to'lovlariga sezilarli darajada ta'sir qiladimi?

# H0: chekish holati individual tibbiy sug'urta to'lovlariga tasir qilmidi
smokers = df[df['smoker'] == 'yes']['charges_log']
non_smokers = df[df['smoker'] == 'no']['charges_log']

t_stat, p_value = stats.ttest_ind(
    non_smokers,
    smokers,
    equal_var=False,
)

print("t-statistic:", t_stat) #-38.86051373676614
print("p-value:", p_value) #1.2314064204246644e-162

if p_value < 0.05:
    print('H0 rejected')
else:
    print('H0 fail to reject')

# H0 rejected

# Xulosa: statistika javoblari boyicha threre is significant difference in insurance charges
# between smokers and non-smokers. Smokerlars have higher charges, which is supported both visually
# (by boxplot) and statistically (T-test). We found there is link between smoking and insurance charges
# but this doesnt mean that smoking causes high charges - they are just related