import numpy as np

a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

# Các phần tử > 5
print("Phần tử > 5:", a[a > 5])

# Vị trí (hàng, cột) của các phần tử > 5
for i, j in zip(*np.where(a > 5)):
    print(f"({i},{j})")