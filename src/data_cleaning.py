import pandas as pd
import os
from config import DATA_PATH, OUTPUT_DIR

def load_and_clean_data():
    print("\nTẢI DỮ LIỆU")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Không tìm thấy dataset!")

    df = pd.read_csv(DATA_PATH)

    print("Kích thước ban đầu:", df.shape)

    # ======================
    # LÀM SẠCH DỮ LIỆU
    # ======================
    print("\nĐANG LÀM SẠCH DỮ LIỆU...")

    # Xóa trùng
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"Số dòng bị trùng đã xóa: {before - df.shape[0]}")

    # Xử lý giá trị thiếu
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("Đã xử lý giá trị thiếu")

    # ======================
    # TẠO ĐẶC TRƯNG
    # ======================
    if 'duration_ms' in df.columns:
        df['duration_min'] = df['duration_ms'] / 60000
        print("Đã tạo cột duration_min")

    # Xử lý decade
    if 'decade' in df.columns:
        df['decade'] = (
            df['decade']
            .astype(str)
            .str.replace('s', '', regex=False)
        )
        df['decade'] = pd.to_numeric(df['decade'], errors='coerce')
        df['decade'] = df['decade'].fillna(0).astype(int) + 1900
        print("Đã chuẩn hóa cột decade")

    # ======================
    # LOẠI BỎ NGOẠI LAI
    # ======================
    def remove_outliers(df, col):
        if col not in df.columns:
            return df

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        before = df.shape[0]

        df = df[
            (df[col] >= Q1 - 1.5 * IQR) &
            (df[col] <= Q3 + 1.5 * IQR)
        ]

        print(f"Số dòng bị loại (outlier) ở {col}: {before - df.shape[0]}")

        return df

    for col in ['tempo', 'loudness']:
        df = remove_outliers(df, col)

    print("Kích thước sau khi làm sạch:", df.shape)

    # ======================
    # LƯU FILE
    # ======================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "spotify_cleaned.csv")

    df.to_csv(output_path, index=False)

    print("Đã lưu dữ liệu tại:", output_path)

    return df