import numpy as np

A = np.array([
    [2, 1],
    [1, 3]
])

B = np.array([
    [4, 2],
    [1, 5]
])

# 1. A + B
print("A + B =\n", A + B)

# 2. A - B
print("A - B =\n", A - B)

# 3. Tích ma trận A @ B
print("A @ B =\n", A @ B)

# 4. Định thức của A
det_A = np.linalg.det(A)
print("det(A) =", det_A)

# 5. Ma trận nghịch đảo của A
inv_A = np.linalg.inv(A)
print("A^-1 =\n", inv_A)

# 6. Giải hệ phương trình: 2x + y = 5, x + 3y = 7
b = np.array([5, 7])
solution = np.linalg.solve(A, b)
print("Nghiệm hệ phương trình:", solution)

# Kiểm tra nghiệm
check = A @ solution
print("Kiểm tra lại:", check)
