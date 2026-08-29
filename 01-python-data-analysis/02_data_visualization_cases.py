import pandas as pd
import matplotlib.pyplot as plt


# 1 case


df = pd.read_csv('data/case1_temperature_line.csv')

fig, ax = plt.subplots()
ax.plot(df['month'], df['Tashkent'], label='Tashkent')
ax.plot(df['month'], df['Dubai'], label='Dubai')
ax.plot(df['month'], df['London'], label='London')

ax.set_ylabel('Temperature')
ax.set_label('Month')
ax.legend()
ax.grid(True)

plt.show()


# 2 case
#
df = pd.read_csv('data/case2_sales_bar.csv')

for i in range(len(df)):
    plt.bar(df['revenue_usd'][i], df['units_sold'][i], label=df['category'][i], width=3000,)

plt.xlabel("Revenue (USD)")
plt.ylabel("Units Sold")
plt.legend(loc="upper center")
plt.grid(True)
plt.show()


# 3 case

df = pd.read_csv('data/case3_study_scatter.csv')

plt.figure(figsize=(10,6))

scatter = plt.scatter(df['study_hours'], df['sleep_hours'], c=df["exam_score"],edgecolors='w', cmap="viridis",alpha=0.8)

plt.colorbar(scatter, label="Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Sleep Hours")
plt.title("Student Data")

plt.show()


# 4 case

df = pd.read_csv('data/case4_ages_histogram.csv')

plt.hist(df['customer_id'], bins=df['customer_id'], weights=df['age'])

plt.ylabel("Number of Customers")
plt.xlabel("Age")
plt.title("Customer Age Distribution")

plt.show()


# 5 case

df = pd.read_csv('data/case5_health_heatmap.csv')

correlation = df.corr()

plt.imshow(correlation)
plt.colorbar()

plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation.columns)), correlation.columns,)

plt.show()

# case 6

df = pd.read_csv('data/case6_salary_boxplot.csv')

departments = df['department'].unique()

data = []

for department in departments:
    data.append(df[df['department'] == department]['salary_usd'])

plt.boxplot(data)

plt.xticks(range(1, len(departments) + 1), departments)
plt.xlabel("Department")
plt.ylabel("Salary (USD)")
plt.show()