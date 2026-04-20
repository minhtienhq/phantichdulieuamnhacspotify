import numpy as np

a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

# Lấy dòng 2 (chỉ số 1 vì bắt đầu từ 0)
print("Dòng 2:", a[1])

# Lấy cột 3 (chỉ số 2)
print("Cột 3:", a[:,2])

# Lấy phần tử (2,2) (hàng 2, cột 2 → chỉ số [1,1])
print("Phần tử (2,2):", a[1,1])