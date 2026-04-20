import pandas as pd

# Đọc file
df = pd.read_csv("D:/Python/bài tập/students_performance.csv")

# Ép kiểu (nếu cần)
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

# Tạo cột Total
df['Total'] = df['Score'] + df['Bonus']

# Chuẩn hóa Score
df['Score_norm'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min())

# In kết quả
print(df[['Score', 'Bonus', 'Total', 'Score_norm']].head())