import pandas as pd

df = pd.read_csv("D:/Python/bài tập/students_performance.csv")

# Ép kiểu + tạo Total (nếu chưa có)
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
df['Total'] = df['Score'] + df['Bonus']

# ===== Theo City =====
print(df.groupby('City')['Score'].agg(['mean', 'count']))

# ===== Theo Class =====
print(df.groupby('Class')['Score'].agg(['mean', 'max']))

# ===== Top 3 Total cao nhất =====
top3 = df.sort_values('Total', ascending=False).head(3)
print(top3[['Name', 'Class', 'Total']])