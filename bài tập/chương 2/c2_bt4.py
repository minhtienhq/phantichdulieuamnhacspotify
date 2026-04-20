import numpy as np

# Tạo ma trận random (5x5)
matrix = np.random.rand(5, 5)   # giá trị ngẫu nhiên từ 0 → 1
print("Ma trận:\n", matrix)

# Tìm max, min, mean, std
print("Max:", matrix.max())
print("Min:", matrix.min())
print("Mean:", matrix.mean())
print("Std:", matrix.std())