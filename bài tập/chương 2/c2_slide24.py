import numpy as np
#Tạo vector có các phần tử là số thực nằm từ 6 đến 10, số phần tử là 5
d=np.random.randint(1,10,size=(5,4))
print("Các phần tử của mảng:")
print(d)
print("Phần tử đầu tiên:",d[0,0])
print("Phần tu thu 2 3:",d[2,3])
print("Phần tử cuối cùng:",d[-1,-1])
print("Cột đầu tiên:",d[:,0])
print("Cột thứ 3:",d[:,2])
print("Dòng đầu tiên:",d[0,:])
print("Dòng thứ 5:",d[4,:])
print("Dòng cuối cùng:",d[-1,:])