import pandas as pd

# Đọc file CSV (ví dụ: data.csv)
df = pd.read_csv("students_cleaned_final.csv")

# Kiểm tra kích thước dữ liệu
print(df.shape)

# Kiểm tra thông tin tổng quan về dữ liệu
print(df.info())