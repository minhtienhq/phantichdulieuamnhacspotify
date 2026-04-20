import pandas as pd
import matplotlib.pyplot as plt

# Đọc file
df = pd.read_csv("D:/Python/bài tập/students_performance.csv")

# Làm sạch City (nên có)
df['City'] = df['City'].fillna('UNKNOWN').str.strip().str.upper()

# Đếm số lượng theo City
city_counts = df['City'].value_counts()

# Vẽ biểu đồ cột
city_counts.plot(kind='bar')

plt.title('So luong sinh vien theo City')
plt.xlabel('City')
plt.ylabel('So luong')
plt.show()