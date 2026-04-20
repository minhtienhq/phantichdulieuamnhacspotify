import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("students_cleaned_final.csv")

# Tạo cột Total
df['Total'] = df['Score'].fillna(0) + df['Bonus'].fillna(0)

# Phân loại Grade
def get_grade(total):
    if total >= 9:
        return 'A'
    elif total >= 8:
        return 'B'
    elif total >= 7:
        return 'C'
    else:
        return 'D'

df['Grade'] = df['Total'].apply(get_grade)

# Đếm số lượng sinh viên theo Grade
grade_counts = df['Grade'].value_counts()

# Vẽ Bar chart
plt.figure(figsize=(8,5))
grade_counts.plot(kind='bar', color='teal', edgecolor='black')
plt.title("Số lượng sinh viên theo Grade")
plt.xlabel("Grade")
plt.ylabel("Số lượng sinh viên")
plt.xticks(rotation=0)
plt.show()