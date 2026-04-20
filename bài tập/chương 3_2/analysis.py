import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PHẦN 1: KIỂM TRA & HIỂU DỮ LIỆU
# =========================
df = pd.read_csv("students_cleaned_final.csv")

print("Shape:", df.shape)
print(df.info())
print("Missing values:\n", df.isnull().sum())
print(df.describe())

# =========================
# PHẦN 2: PHÂN TÍCH TỔNG QUAN
# =========================
city_counts = df['City'].value_counts()
class_counts = df['Class'].value_counts()

city_counts.plot(kind='bar', title="Số lượng sinh viên theo City")
plt.show()

class_counts.plot(kind='bar', title="Số lượng sinh viên theo Class")
plt.show()

# =========================
# PHẦN 3: PHÂN TÍCH ĐIỂM SỐ
# =========================
df['Score'].plot(kind='hist', bins=20, title="Phân bố điểm")
plt.xlabel("Score")
plt.show()

# =========================
# PHẦN 4: SO SÁNH NHÓM
# =========================
mean_score_city = df.groupby('City')['Score'].mean()
mean_score_city.plot(kind='bar', title="Điểm trung bình theo City")
plt.show()

# Tạo cột Total = Score + Bonus
df['Total'] = df['Score'] + df['Bonus']

mean_total_class = df.groupby('Class')['Total'].mean()
mean_total_class.plot(kind='bar', title="Tổng điểm trung bình theo Class")
plt.show()

# =========================
# PHẦN 5: PHÂN TÍCH MỐI QUAN HỆ
# =========================
plt.scatter(df['Age'], df['Score'])
plt.xlabel("Age")
plt.ylabel("Score")
plt.title("Tuổi vs Điểm số")
plt.show()

# =========================
# PHẦN 6: PHÂN TÍCH HỌC LỰC
# =========================
# Tạo cột Grade dựa trên Score
def classify_grade(score):
    if pd.isnull(score):
        return "Unknown"
    elif score >= 9:
        return "Giỏi"
    elif score >= 7:
        return "Khá"
    elif score >= 5:
        return "Trung bình"
    else:
        return "Yếu"

df['Grade'] = df['Score'].apply(classify_grade)

# Thống kê số lượng theo Grade
grade_counts = df['Grade'].value_counts()

grade_counts.plot(kind='bar', title="Phân bố học lực")
plt.show()

grade_counts.plot(kind='pie', autopct='%1.1f%%', title="Phân bố học lực")
plt.ylabel("")
plt.show()