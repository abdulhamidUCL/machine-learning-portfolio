import numpy as np
import pandas as pd


# 1 NumPy: 20 ta sondan massiv yarating va statistikani hisoblang
massiv = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16, 17, 18, 19, 20])
print(massiv.ndim) # nechta dimension bor
print(massiv.shape) # nechiga nechi 20 X 1
print(massiv.dtype) # int
print(massiv.size) # nechta bor ichida

print(massiv.max())
print(massiv.min())

print(massiv.mean()) # average
print(massiv.std())
print(massiv.sum())

# 2 Pandas: o'zingiz haqingizda DataFrame yarating (kamida 5 ustun)
data = {
    'name': 'Abdulhamid',
    'family name': 'Solikjonov',
    'age': 18,
    'kurs': 2,
    'hobby': 'chess',
}

df = pd.DataFrame([data])
print(df)


# 3 CSV faylni o'qing va describe() natijasini tahlil qiling
df = pd.read_csv('talabalar.csv')
print(df.describe())


# 4 groupby() bilan bitta savol toping va javobini yozing
df = pd.read_csv('talabalar.csv')

Savol = 'Toshkentda nechta talaba bor'
print(
    df.groupby([df['shahar'] == 'Toshkent']).size()
)

Javob = '9 ta oqidi, 91 bitta oqimidi'

# 5 Kaggle bilan tanishib chiqing va dataset yuklab darsda o'rgangan bilimlaringizni qo'llang
df = pd.read_csv('talabalar.csv')

print(df.head()) # birinchi 5 ta
print(df.tail()) # ohirgi 5 ta
print(df.describe()) # statistikasi
print(df[['ism', 'shahar']]) # faqat ism bilan shahar ustunlarini chiqaradi
print(df.isnull().sum()) # nechta boshlari bor
print(df.sort_values(by='ism', ascending=False)) # ismlani z-a gacha
print(df.value_counts())# Name: count, Length: 100, dtype: int64 shu narsa chiqdi