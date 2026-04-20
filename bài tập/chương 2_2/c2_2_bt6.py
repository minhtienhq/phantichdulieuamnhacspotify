import pandas as pd

# Đọc file
df = pd.read_csv("D:/Python/bài tập/students_performance.csv")

# Lọc dữ liệu bất thường
outliers = df[(df['Score'] > 9.5) | (df['Score'] < 5)]

# In ra
print(outliers)