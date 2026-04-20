import numpy as np
#Tạo vector có các phần tử là số thực nằm từ 6 đến 10,
số phần tử là 5
d=np.linspace(6,10,5)
print(d)
#Chuyển kiểu float sang int
a=d.astype(np.int16) #d.astype(int)
print(a)
print("Kieu mới:",a.dtype)
#chuyển kiểu int sang unicode
s=a.astype(np.str_)
print("Kieu mới:",s.dtype)
#chuyển từ kiểu int sang bool
b=a.astype(np.bool_)
print("Kieu mới:",b.dtype)