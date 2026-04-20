import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("students_cleaned_final.csv")

# Thống kê số lượng sinh viên theo City
city_counts = df['City'].value_counts()
print("Số lượng sinh viên theo City:")
print(city_counts)

# Thống kê số lượng sinh viên theo Class
class_counts = df['Class'].value_counts()
print("\nSố lượng sinh viên theo Class:")
print(class_counts)