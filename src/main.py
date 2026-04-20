from data_cleaning import load_and_clean_data
from eda import run_eda
from visualization import plot_all
from modeling import train_model
from insight import show_insight

print("\nBẮT ĐẦU CHẠY PIPELINE")

# ======================
# 1. LÀM SẠCH DỮ LIỆU
# ======================
df = load_and_clean_data()

# ======================
# 2. PHÂN TÍCH EDA
# ======================
numeric_cols, corr, top_artists, decade_popularity = run_eda(df)

# ======================
# 3. CHỌN FEATURE CHUẨN
# ======================
target = 'popularity'

features = [
    col for col in numeric_cols
    if col not in ['popularity', 'duration_ms']
]

print("\nFEATURES DÙNG CHO MODEL:")
print(features)

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
    df,
    corr=corr,
    top_artists=top_artists,
    decade_popularity=decade_popularity,
    y_test=y_test,
    y_pred=y_pred,
    y_pred_rf=y_pred_rf
)

# ======================
# 6. INSIGHT
# ======================
if decade_popularity is not None:
    show_insight(decade_popularity)
else:
    print("Không có dữ liệu để tạo insight")

print("\nHOÀN THÀNH TOÀN BỘ PIPELINE")