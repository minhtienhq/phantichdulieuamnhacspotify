import pandas as pd

df = pd.read_csv("D:/Python/bài tập/students_performance.csv")

# Làm sạch Name
df['Name'] = df['Name'].str.strip().str.title()

# Làm sạch City
df['City'] = df['City'].str.strip().str.upper()

print(df[['Name','City']].head())