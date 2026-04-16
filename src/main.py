from data_cleaning import load_and_clean_data
from eda import run_eda
from visualization import plot_all
from modeling import train_model
from insight import show_insight

import pandas as pd
from sklearn.preprocessing import StandardScaler

print("\nBẮT ĐẦU CHẠY PIPELINE")

# ======================
# 1. LÀM SẠCH DỮ LIỆU
# ======================
df = load_and_clean_data()

# ======================
# 2. PHÂN TÍCH EDA
# ======================
features, top_artists, decade_popularity = run_eda(df)

# ======================
# 3. TÍNH CORRELATION
# ======================
if features:
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[features]),
        columns=features
    )
    corr = df_scaled.corr()
    print("Đã tính ma trận tương quan")
else:
    corr = None
    print("Không có feature để tính correlation")

# ======================
# 4. HUẤN LUYỆN MODEL
# ======================
if features:
    y_test, y_pred, y_pred_rf = train_model(df, features)
else:
    print("Bỏ qua bước modeling vì không có feature")
    y_test, y_pred, y_pred_rf = None, None, None

# ======================
# 5. VẼ BIỂU ĐỒ
# ======================
plot_all(
    df, corr, top_artists, decade_popularity,
    y_test, y_pred, y_pred_rf
)

# ======================
# 6. INSIGHT
# ======================
if decade_popularity is not None:
    show_insight(decade_popularity)
else:
    print("Không có dữ liệu để tạo insight")

print("\nHOÀN THÀNH TOÀN BỘ PIPELINE")