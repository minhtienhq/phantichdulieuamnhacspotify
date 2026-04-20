import pandas as pd

# Đọc file CSV
df = pd.read_csv("D:/Python/bài tập/students_performance.csv")

# Hiển thị 5 dòng đầu
print("=== 5 dòng đầu ===")
print(df.head())

# Hiển thị thông tin dữ liệu
print("\n=== Thông tin dữ liệu ===")
df.info()

# Hiển thị thống kê mô tả
print("\n=== Thống kê mô tả ===")
print(df.describe())