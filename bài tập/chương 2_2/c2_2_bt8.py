import pandas as pd

df = pd.read_csv("D:/Python/bài tập/students_performance.csv")

# Xử lý cơ bản
df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].mean())
df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(df['Score'].mean())
df['City'] = df['City'].fillna('UNKNOWN').str.strip().str.upper()
df['Name'] = df['Name'].str.strip().str.title()
df = df.drop_duplicates()

# ===== In kết quả =====
print("=== DataFrame sau xử lý ===")
print(df)

print("\n=== Mean Score theo City ===")
print(df.groupby('City')['Score'].mean())

print("\n=== Số lượng theo City ===")
print(df['City'].value_counts())