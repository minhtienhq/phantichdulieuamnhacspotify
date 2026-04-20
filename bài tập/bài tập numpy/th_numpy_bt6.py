import pandas as pd
import numpy as np

# =======================
# 1. TẠO DỮ LIỆU BAN ĐẦU
# =======================
data = {
    "MaSV": ["SV01", "SV02", "SV03", "SV03", "SV05", "SV06", "SV07", "SV08"],
    "Tuoi": [20, 21, 19, 19, None, 22, 35, 20],
    "GioiTinh": ["Nam", "Nữ", "nu", "nu", "Nam", "Nữ", "Nam", None],
    "GioTuHoc": [2.5, 3, None, 4, 2, 10, -1, 3.5],
    "GioMangXaHoi": [4, 5, 3.5, 3.5, 20, 2, 5, None],
    "DiemTB": [3.1, 2.8, 3.5, 3.5, 2.0, 3.8, 4.5, None]
}

df = pd.DataFrame(data)

print("=== DỮ LIỆU BAN ĐẦU ===")
print(df)
print("\nKích thước dữ liệu:", df.shape)

# =======================
# 2. KIỂM TRA GIÁ TRỊ THIẾU
# =======================
print("\n=== GIÁ TRỊ THIẾU ===")
print(df.isnull().sum())

# =======================
# 3. XÓA TRÙNG LẶP
# =======================
df = df.drop_duplicates(subset="MaSV")

# =======================
# 4. CHUẨN HÓA GIỚI TÍNH
# =======================
df["GioiTinh"] = df["GioiTinh"].replace({
    "nu": "Nữ",
    "Nữ": "Nữ",
    "Nam": "Nam"
})
df["GioiTinh"] = df["GioiTinh"].fillna("Không rõ")

# =======================
# 5. XỬ LÝ GIÁ TRỊ THIẾU
# =======================
df["Tuoi"] = df["Tuoi"].fillna(df["Tuoi"].mean())
df["GioTuHoc"] = df["GioTuHoc"].fillna(df["GioTuHoc"].mean())
df["GioMangXaHoi"] = df["GioMangXaHoi"].fillna(df["GioMangXaHoi"].mean())
df["DiemTB"] = df["DiemTB"].fillna(df["DiemTB"].mean())

# =======================
# 6. XỬ LÝ NGOẠI LỆ
# =======================
df.loc[df["Tuoi"] > 30, "Tuoi"] = df["Tuoi"].mean()
df.loc[df["GioTuHoc"] < 0, "GioTuHoc"] = df["GioTuHoc"].mean()
df.loc[df["GioMangXaHoi"] > 12, "GioMangXaHoi"] = df["GioMangXaHoi"].mean()
df.loc[df["DiemTB"] > 4.0, "DiemTB"] = df["DiemTB"].mean()

# =======================
# 7. DỮ LIỆU SAU LÀM SẠCH
# =======================
print("\n=== DỮ LIỆU SAU LÀM SẠCH ===")
print(df)

# =======================
# 8. CHUẨN HÓA MIN-MAX
# =======================
cols = ["Tuoi", "GioTuHoc", "GioMangXaHoi", "DiemTB"]

df_minmax = df.copy()
for col in cols:
    df_minmax[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

print("\n=== DỮ LIỆU MIN-MAX ===")
print(df_minmax)

# =======================
# 9. CHUẨN HÓA Z-SCORE
# =======================
df_zscore = df.copy()
for col in cols:
    df_zscore[col] = (df[col] - df[col].mean()) / df[col].std()

print("\n=== DỮ LIỆU Z-SCORE ===")
print(df_zscore)

# =======================
# 10. SO SÁNH NHANH
# =======================
print("\n=== NHẬN XÉT ===")
print("- Min-Max: Đưa dữ liệu về khoảng [0,1]")
print("- Z-score: Chuẩn hóa theo trung bình = 0, độ lệch chuẩn = 1")
print("- Sau chuẩn hóa: dữ liệu đồng nhất hơn, dễ phân tích và mô hình hóa")