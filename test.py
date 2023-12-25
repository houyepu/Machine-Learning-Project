import pandas as pd

# Sample data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Ethan'],
    'Age': [25, 32, 45, 20, 35],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
}

# Creating DataFrame
df = pd.DataFrame(data)

# Adding a new column 'Over 30'
df['Over 30'] = df['Age'] > 30

# Display the DataFrame
print(df)