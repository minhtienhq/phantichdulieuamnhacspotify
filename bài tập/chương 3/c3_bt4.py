import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("students_cleaned_final.csv")

# Vẽ Histogram cho cột Score
plt.figure(figsize=(8,5))
plt.hist(df['Score'], bins=10, color='skyblue', edgecolor='black')
plt.title("Phân bố điểm số (Histogram)")
plt.xlabel("Score")
plt.ylabel("Số lượng sinh viên")
plt.show()