import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("students_cleaned_final.csv")

# Vẽ Scatter plot Age vs Score
plt.figure(figsize=(8,5))
plt.scatter(df['Age'], df['Score'], color='blue', alpha=0.6, edgecolors='black')
plt.title("Mối quan hệ giữa Age và Score")
plt.xlabel("Age")
plt.ylabel("Score")
plt.show()