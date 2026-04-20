import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("students_cleaned_final.csv")

# Tính điểm trung bình theo City
mean_scores = df.groupby('City')['Score'].mean()

# Vẽ Bar chart
plt.figure(figsize=(8,5))
mean_scores.plot(kind='bar', color='orange', edgecolor='black')
plt.title("Mean Score theo City")
plt.xlabel("City")
plt.ylabel("Điểm trung bình (Mean Score)")
plt.xticks(rotation=45)
plt.show()