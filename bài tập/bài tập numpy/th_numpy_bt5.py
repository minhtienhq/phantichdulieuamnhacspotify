import numpy as np
import matplotlib.pyplot as plt

# 1. Tạo 100 bước ngẫu nhiên (+1 hoặc -1)
np.random.seed(42)  # để kết quả tái lập
steps = np.random.choice([-1, 1], size=100)

# 2. Tính vị trí sau mỗi bước (cộng dồn)
walk = np.cumsum(steps)

# 3. In 10 giá trị đầu tiên
print("10 vị trí đầu tiên:", walk[:10])

# 4. Vẽ đồ thị random walk
plt.plot(walk)
plt.title("Random Walk 1 chiều")
plt.xlabel("Bước")
plt.ylabel("Vị trí")
plt.grid(True)
plt.show()

# 5. Vị trí cuối cùng, lớn nhất, nhỏ nhất
print("Vị trí cuối cùng:", walk[-1])
print("Vị trí lớn nhất:", np.max(walk))
print("Vị trí nhỏ nhất:", np.min(walk))
