import numpy as np
scores = np.array([
    [7.5, 8.0, 6.5, 9.0],
    [6.0, 7.0, 7.5, 8.0],
    [8.5, 9.0, 8.0, 9.5],
    [5.5, 6.0, 6.5, 7.0],
    [9.0, 8.5, 9.5, 8.0]
])
# 1. In ra ma trận điểm
print("Ma trận điểm:\n", scores)

# 2. Điểm trung bình toàn bộ ma trận
print("Điểm trung bình toàn bộ:", np.mean(scores))

# 3. Điểm trung bình theo từng sinh viên (axis=1)
print("Điểm trung bình từng sinh viên:", np.mean(scores, axis=1))

# 4. Điểm trung bình theo từng môn (axis=0)
print("Điểm trung bình từng môn:", np.mean(scores, axis=0))

# 5. Điểm cao nhất và thấp nhất trong ma trận
print("Điểm cao nhất:", np.max(scores))
print("Điểm thấp nhất:", np.min(scores))

# 6. Độ lệch chuẩn theo từng môn
print("Độ lệch chuẩn từng môn:", np.std(scores, axis=0))

# 7. Sinh viên có điểm trung bình cao nhất
avg_students = np.mean(scores, axis=1)
best_student = np.argmax(avg_students)
print("Sinh viên có điểm TB cao nhất là vị trí:", best_student)
