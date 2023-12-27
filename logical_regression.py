import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)

iris = pd.read_csv("IRIS.csv")
print(iris.head())
iris["variety"].value_counts()
