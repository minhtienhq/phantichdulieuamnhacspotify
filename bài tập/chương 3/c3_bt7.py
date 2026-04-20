import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("students_cleaned_final.csv")

# Tạo cột Total = Score + Bonus
df['Total'] = df['Score'].fillna(0) + df['Bonus'].fillna(0)

# Tính Mean Total theo Class
mean_total = df.groupby('Class')['Total'].mean()

print(mean_total)

# Vẽ Bar chart
plt.figure(figsize=(8,5))
mean_total.plot(kind='bar', color='purple', edgecolor='black')
plt.title("Mean Total theo Class")
plt.xlabel("Class")
plt.ylabel("Điểm trung bình (Mean Total)")
plt.xticks(rotation=45)
plt.show()