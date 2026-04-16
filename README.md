Spotify Data Analysis & Machine Learning Project
📌 Giới thiệu

Dự án này xây dựng một hệ thống phân tích dữ liệu Spotify end-to-end, bao gồm:

🧹 Làm sạch dữ liệu (Data Cleaning)
🔍 Phân tích khám phá dữ liệu (EDA - Exploratory Data Analysis)
📊 Trực quan hóa dữ liệu
🤖 Huấn luyện mô hình Machine Learning
📈 Dự đoán độ phổ biến (popularity) của bài hát
💡 Sinh insight tự động
🎯 Mục tiêu
Hiểu đặc điểm của các bài hát trên Spotify
Phân tích xu hướng âm nhạc theo nghệ sĩ và thập niên
Xây dựng mô hình dự đoán độ phổ biến của bài hát
Rút ra insight phục vụ phân tích kinh doanh / âm nhạc
⚙️ Công nghệ sử dụng
Python 3.x
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
Joblib
📂 Cấu trúc thư mục
project/
│── data/
│   └── spotify_dataset.csv
│
│── output/
│   ├── spotify_cleaned.csv
│   ├── top_artists.csv
│   ├── decade_popularity.csv
│   ├── eda_stats.csv
│   ├── correlation.csv
│   ├── feature_importance.csv
│   ├── predictions.csv
│   └── *.png
│
│── models/
│   ├── pipeline_lr.pkl
│   └── pipeline_rf.pkl
│
│── config.py
│── data_cleaning.py
│── eda.py
│── visualization.py
│── modeling.py
│── insight.py
│── main.py
⚙️ Chức năng chính
1. ⚙️ config.py
Quản lý đường dẫn:
Dataset
Output
Models
Tự động tạo thư mục nếu chưa tồn tại
2. 🧹 data_cleaning.py

Hàm chính: load_and_clean_data()

Thực hiện:

Đọc dữ liệu từ CSV
Xóa dữ liệu trùng
Xử lý missing value (median)
Feature engineering:
duration_min từ duration_ms
Chuẩn hóa decade
Loại bỏ outliers (IQR):
tempo
loudness

📁 Output:

output/spotify_cleaned.csv
3. 🔍 eda.py

Hàm chính: run_eda(df)

Thực hiện:

Chọn feature quan trọng:
danceability, energy, tempo, valence, loudness
Phân tích:
Top 10 nghệ sĩ
Popularity theo thập niên
Thống kê mô tả
Ma trận tương quan

📁 Output:

top_artists.csv
decade_popularity.csv
eda_stats.csv
correlation.csv
4. 📊 visualization.py

Sinh các biểu đồ:

Histogram (popularity)
Heatmap correlation
Top artists
Trend theo thập niên
Scatter: danceability vs popularity
Boxplot
Residual plot
So sánh mô hình

📁 Output:

*.png
5. 🤖 modeling.py

Huấn luyện 2 mô hình:

Linear Regression
Random Forest

Đánh giá bằng:

R2 Score
MSE
MAE

📁 Output:

models/
predictions.csv
feature_importance.csv
model_comparison.png
6. 💡 insight.py

Sinh insight tự động:

Thập niên phổ biến nhất / thấp nhất
Xu hướng tăng/giảm
Nhận xét về âm nhạc

📁 Output:

output/insight.txt
7. 🚀 main.py

Chạy toàn bộ pipeline:

from data_cleaning import load_and_clean_data
from eda import run_eda
from visualization import plot_all
from modeling import train_model
from insight import show_insight

def main():
    df = load_and_clean_data()
    features, top_artists, decade_popularity = run_eda(df)
    y_test, y_pred, y_pred_rf = train_model(df, features)
    plot_all(df, None, top_artists, decade_popularity,
             y_test, y_pred, y_pred_rf)
    show_insight(decade_popularity)

if __name__ == "__main__":
    main()
🚀 Cách chạy dự án
1. Cài thư viện
pip install pandas numpy scikit-learn matplotlib seaborn joblib
2. Chạy chương trình
python main.py
📊 Kết quả đầu ra

Sau khi chạy, bạn sẽ có:

✅ Dữ liệu đã làm sạch
✅ Top nghệ sĩ
✅ Xu hướng theo thập niên
✅ Thống kê dữ liệu
✅ Ma trận tương quan
✅ Biểu đồ trực quan
✅ Dự đoán ML
✅ Insight tự động
💡 Insight mẫu
Danceability cao → dễ viral
Tempo ~120 BPM phổ biến
Nhạc mới có lợi thế trên nền tảng streaming
📌 Ghi chú
Dataset cần đặt tại:
data/spotify_dataset.csv
Output sẽ tự động lưu vào:
output/
Pipeline có khả năng mở rộng:
Deep Learning
API (FastAPI / Flask)
Dashboard (Streamlit)
👤 Tác giả
Your Name
⭐ Gợi ý cải tiến
Hyperparameter tuning
Feature engineering nâng cao
Deploy model
Xây dashboard trực quan