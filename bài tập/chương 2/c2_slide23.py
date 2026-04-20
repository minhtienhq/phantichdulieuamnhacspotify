import numpy as np
#Tạo vector có các phần tử là số thực nằm từ 6 đến 10, số phần tử là 5
d=np.arange(6,40,step=2)
print("Các phần tử của mảng:",d)
print("Phần tử đầu tiên:",d[0])
print("Phần tử cuối cùng:",d[-1])
print("5 phần tử đầu tiên:",d[:5])
print("các phần tử từ 5 đến 10",d[5:10])
print("các phần tử từ 10 đến hết",d[10:])