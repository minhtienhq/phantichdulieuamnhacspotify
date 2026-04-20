import numpy as np

scores = np.array([
    [7.5, 8.0, 6.5, 9.0],
    [6.0, 7.0, 7.5, 8.0],
    [8.5, 9.0, 8.0, 9.5],
    [5.5, 6.0, 6.5, 7.0],
    [9.0, 8.5, 9.5, 8.0]
])

# 1. Vector trung bình từng môn
mean_col = np.mean(scores, axis=0)
print("Trung bình từng môn:", mean_col)

# 2. Vector độ lệch chuẩn từng môn
std_col = np.std(scores, axis=0)
print("Độ lệch chuẩn từng môn:", std_col)

# 3. Chuẩn hóa toàn bộ ma trận bằng Z-score (broadcasting)
z_scores = (scores - mean_col) / std_col

# 4. In ma trận đã chuẩn hóa, làm tròn 2 chữ số thập phân
print("Ma trận chuẩn hóa (Z-score):\n", np.round(z_scores, 2))

# 5. Kiểm tra lại trung bình các cột sau chuẩn hóa
print("Trung bình các cột sau chuẩn hóa:", np.mean(z_scores, axis=0))