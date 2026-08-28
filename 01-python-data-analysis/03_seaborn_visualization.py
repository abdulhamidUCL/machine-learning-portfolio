import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris = sns.load_dataset("iris")
titanic = sns.load_dataset("titanic")
# print(titanic)

sns.set_theme(style="darkgrid",
              palette="deep",
              font_scale=1,
              )


# 1 scatterplot
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=iris,
    x="sepal_length",
    y="sepal_width",
    hue="species",
    style="species",
    s=80
)


plt.show()


# 2 histplot
plt.figure(figsize=(10,6))

sns.histplot(
    data=titanic,
    x='age',
    bins=20,
    kde=True,
)

plt.show()


# 3 kdeplot
plt.figure(figsize=(10,6))

with sns.axes_style("whitegrid"):
    sns.kdeplot(
        data=titanic,
        x='age',
        fill=True,
        bw_adjust=0.6
    )
plt.title('Density of ages')

plt.show()

# 4 boxplot
plt.figure(figsize=(10, 10))

sns.boxplot(
    data=iris,
    x="species",
    y="petal_width",
)
plt.yticks(np.arange(0, 3, 0.05))
plt.title("Petal Width by Species")
plt.xlabel("Species")
plt.ylabel("Petal Width (cm)")

plt.show()


# 5 violinplot
plt.figure(figsize=(10, 6))

sns.violinplot(
    data=titanic,
    x="who",
    y="age",
)

plt.show()


# 6 stripplot
plt.figure(figsize=(10,6))

sns.stripplot(
    data=titanic,
    x='who',
    y='age',
    jitter=True,
    alpha=0.7,
)

plt.show()

# 7 swarmplot
plt.figure(figsize=(10,6))

sns.swarmplot(
    data=titanic,
    x='who',
    y='age',
)

plt.show()

# 8 regplot
plt.figure(figsize=(10,6))

sns.regplot(
    data=iris,
    x='sepal_length',
    y='sepal_width',
    ci=95,
)

plt.show()

# 9 joinplot
plt.figure(figsize=(10,6))

sns.jointplot(
    data=iris,
    x='sepal_length',
    y='sepal_width',
)

plt.show()