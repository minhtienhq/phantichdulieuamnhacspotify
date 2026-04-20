import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("students_cleaned_final.csv")

# Thống kê số lượng sinh viên theo City
city_counts = df['City'].value_counts()

# Vẽ bar chart cho City
plt.figure(figsize=(8,5))
city_counts.plot(kind='bar', color='skyblue')
plt.title("Số lượng sinh viên theo City")
plt.xlabel("City")
plt.ylabel("Số lượng sinh viên")
plt.xticks(rotation=45)
plt.show()

# Thống kê số lượng sinh viên theo Class
class_counts = df['Class'].value_counts()

# Vẽ bar chart cho Class
plt.figure(figsize=(8,5))
class_counts.plot(kind='bar', color='lightgreen')
plt.title("Số lượng sinh viên theo Class")
plt.xlabel("Class")
plt.ylabel("Số lượng sinh viên")
plt.xticks(rotation=45)
plt.show()