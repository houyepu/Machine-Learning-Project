import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load the data from the uploaded CSV file
file_path = 'Machine-Learning-Project\Salary.csv'
salary_df = pd.read_csv(file_path)

# Prepare the data
X = salary_df[['YearsExperience']]
y = salary_df['Salary']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions for plotting
y_pred = model.predict(X)

# Create the regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x='YearsExperience', y='Salary', data=salary_df, scatter_kws={"s": 50}, order=1, ci=None, scatter=True)
plt.plot(X, y_pred, color='red') # regression line
plt.title('Salary vs Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
