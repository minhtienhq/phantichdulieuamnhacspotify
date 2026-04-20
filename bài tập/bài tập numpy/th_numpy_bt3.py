import numpy as np

# Số lượng bán ra của 3 sản phẩm trong 4 ngày
quantity = np.array([
    [10, 12, 9, 14],
    [5, 7, 8, 6],
    [20, 18, 25, 22]
])

# Giá bán từng sản phẩm
price = np.array([15000, 25000, 10000])

# 1. Doanh thu của từng sản phẩm theo từng ngày
revenue = quantity * price.reshape(3, 1)
print("Doanh thu từng sản phẩm theo ngày:\n", revenue)

# 2. Tổng doanh thu của từng sản phẩm
sum_product = np.sum(revenue, axis=1)
print("Tổng doanh thu từng sản phẩm:", sum_product)

# 3. Tổng doanh thu của từng ngày
sum_day = np.sum(revenue, axis=0)
print("Tổng doanh thu từng ngày:", sum_day)

# 4. Ngày có doanh thu cao nhất
best_day = np.argmax(sum_day) + 1  # +1 để tính theo ngày 1-4
print("Ngày doanh thu cao nhất:", best_day)

# 5. Tỷ trọng doanh thu của từng sản phẩm trong toàn bộ doanh thu
ratio = sum_product / np.sum(sum_product)
print("Tỷ trọng doanh thu:", np.round(ratio * 100, 2), "%")