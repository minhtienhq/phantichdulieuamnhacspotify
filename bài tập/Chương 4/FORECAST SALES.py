# ======================================
# 1. IMPORT LIBRARIES
# ======================================
# Thư viện xử lý dữ liệu
import pandas as pd
import numpy as np

# Thư viện vẽ biểu đồ (EDA + Visualization)
import matplotlib.pyplot as plt

# Thư viện Machine Learning
from sklearn.linear_model import LinearRegression      # Mô hình Linear Regression (bắt buộc)
from sklearn.ensemble import RandomForestRegressor     # Mô hình Random Forest
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Đánh giá mô hình

# ======================================
# 2. LOAD DATA
# ======================================
# Đọc file dữ liệu 3 năm gần nhất
# Dataset gồm: date, sales, promotion_budget, num_customers,...
df = pd.read_csv("D:/Python/bài tập/dataset.csv")

# ======================================
# 3. DATA PREPARATION (Bước 1 trong yêu cầu)
# ======================================

# Convert cột date → datetime (quan trọng cho time series)
df['date'] = pd.to_datetime(df['date'])

# Sắp xếp theo thời gian
df = df.sort_values('date')

# Set date làm index (time index)
df.set_index('date', inplace=True)

# Xử lý dữ liệu thiếu và trùng
df = df.drop_duplicates()   # Xóa duplicate
df = df.ffill()             # Điền missing values (forward fill)

# Resample về dữ liệu theo tháng (đúng format cuối tháng)
df = df.resample('ME').sum()

# Đảm bảo index là ngày cuối tháng
df.index = df.index.to_period('M').to_timestamp('M')

# ===== Feature Engineering =====
# Tạo thêm biến thời gian
df['month'] = df.index.month        # Tháng
df['quarter'] = df.index.quarter    # Quý

# Tạo biến trễ (lag features)
df['lag_1'] = df['sales'].shift(1)   # Doanh thu tháng trước
df['lag_3'] = df['sales'].shift(3)   # Doanh thu 3 tháng trước

# Trung bình trượt 3 tháng
df['rolling_mean_3'] = df['sales'].rolling(3).mean()

# Xóa các dòng bị NaN sau khi tạo lag
df.dropna(inplace=True)

# ======================================
# 4. EDA – KHÁM PHÁ DỮ LIỆU (Bước 2 + 3)
# ======================================

# 1. Line chart: Doanh thu theo thời gian
plt.figure()
plt.plot(df.index, df['sales'])
plt.title("Sales over time")
plt.show()

# 2. Rolling mean (xu hướng)
plt.figure()
df['sales'].rolling(3).mean().plot(label='MA 3')
df['sales'].rolling(6).mean().plot(label='MA 6')
plt.legend()
plt.title("Rolling Mean")
plt.show()

# 3. Bar chart theo tháng (kiểm tra seasonality)
plt.figure()
df.groupby('month')['sales'].mean().plot(kind='bar')
plt.title("Sales by Month")
plt.show()

# 4. Scatter: Promotion vs Sales (tương quan marketing)
plt.figure()
plt.scatter(df['promotion_budget'], df['sales'])
plt.title("Promotion vs Sales")
plt.xlabel("Promotion Budget")
plt.ylabel("Sales")
plt.show()

# ======================================
# 5. TRAIN TEST SPLIT
# ======================================
# X = feature, y = target (sales)
X = df.drop('sales', axis=1)
y = df['sales']

# Chia dữ liệu theo thời gian (80% train, 20% test)
train_size = int(len(df) * 0.8)

X_train = X[:train_size]
X_test  = X[train_size:]

y_train = y[:train_size]
y_test  = y[train_size:]

# ======================================
# 6. MODELING (Bước 4)
# ======================================

# ----- Model 1: Linear Regression (bắt buộc) -----
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# ----- Model 2: Random Forest -----
model_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# ----- Baseline: Moving Average -----
y_pred_ma = df['sales'].rolling(3).mean().shift(1)
y_pred_ma = y_pred_ma[-len(y_test):]

# ======================================
# 7. EVALUATION (Bước 5)
# ======================================

# Hàm đánh giá
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)          # Sai số tuyệt đối
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # Sai số bình phương
    return mae, rmse

# Tính toán
mae_lr, rmse_lr = evaluate(y_test, y_pred_lr)
mae_rf, rmse_rf = evaluate(y_test, y_pred_rf)
mae_ma, rmse_ma = evaluate(y_test, y_pred_ma)

# In kết quả
print("=== MODEL PERFORMANCE ===")
print(f"Linear Regression  -> MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}")
print(f"Random Forest      -> MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")
print(f"Moving Average     -> MAE: {mae_ma:.2f}, RMSE: {rmse_ma:.2f}")

# ======================================
# 8. CHỌN MODEL TỐT NHẤT
# ======================================
# Chọn model có RMSE nhỏ hơn
best_model = model_rf if rmse_rf < rmse_lr else model_lr
print("\nBest model:", "Random Forest" if best_model == model_rf else "Linear Regression")

# ======================================
# 9. FORECAST 12 THÁNG (Bước 6)
# ======================================
future_preds = []
last_data = df.copy()

for i in range(12):
    last_row = last_data.iloc[-1:].copy()

    # Predict doanh thu tháng tiếp theo
    X_last = last_row.drop('sales', axis=1)
    pred = best_model.predict(X_last)[0]
    future_preds.append(pred)

    # Tạo tháng mới
    new_date = last_data.index[-1] + pd.DateOffset(months=1)

    # Tạo dòng dữ liệu mới
    new_row = last_row.copy()
    new_row.index = [new_date]
    new_row['sales'] = pred

    # Update feature
    new_row['month'] = new_date.month
    new_row['quarter'] = new_date.quarter

    new_row['lag_1'] = last_data['sales'].iloc[-1]
    new_row['lag_3'] = last_data['sales'].iloc[-3]
    new_row['rolling_mean_3'] = last_data['sales'].iloc[-3:].mean()

    # Giả định: marketing và khách giữ nguyên
    new_row['promotion_budget'] = last_row['promotion_budget'].values[0]
    new_row['num_customers'] = last_row['num_customers'].values[0]

    last_data = pd.concat([last_data, new_row])

# ======================================
# 10. PLOT FORECAST
# ======================================
future_dates = pd.date_range(start=df.index[-1], periods=13, freq='ME')[1:]

plt.figure()
plt.plot(df.index, df['sales'], label='Actual')
plt.plot(future_dates, future_preds, label='Forecast', linestyle='--')
plt.legend()
plt.title("Sales Forecast (12 months)")
plt.show()

# ======================================
# 11. OUTPUT
# ======================================
print("\nForecast next 12 months:")
for date, value in zip(future_dates, future_preds):
    print(f"{date.date()} : {value:.2f}")