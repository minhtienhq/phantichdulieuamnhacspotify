import pandas as pd

# ===== 1. Đọc file =====
df = pd.read_csv("D:/Python/bài tập/students_performance.csv")

# ===== 2. Ép kiểu dữ liệu =====
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

# ===== 3. Xử lý missing =====
# Age → trung bình
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Score → trung bình
df['Score'] = df['Score'].fillna(df['Score'].mean())

# City → UNKNOWN
df['City'] = df['City'].fillna('UNKNOWN')

# ===== 4. Xóa dòng trùng =====
df = df.drop_duplicates()

# ===== 5. Kiểm tra lại =====
print("=== Dữ liệu thiếu ===")
print(df.isnull().sum())

print("\n=== Số dòng trùng ===")
print(df.duplicated().sum())

print("\n=== Dữ liệu sau xử lý ===")
print(df.head())