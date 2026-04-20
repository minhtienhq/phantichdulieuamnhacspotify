import pandas as pd

# Đọc file
df = pd.read_csv("D:/Python/bài tập/students_performance.csv")

# (có thể xử lý nếu cần)
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

# Lưu file
df.to_csv(r'D:\Python\students_cleaned_final.csv', index=False)