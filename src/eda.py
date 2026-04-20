import os
import pandas as pd
from config import OUTPUT_DIR


def run_eda(df):
    print("\nPHÂN TÍCH DỮ LIỆU (EDA)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ======================
    # 1. CHỌN CỘT SỐ TỰ ĐỘNG
    # ======================
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    print("\nCác cột số:")
    print(numeric_cols)

    # ======================
    # 2. KIỂM TRA NULL
    # ======================
    print("\nGIÁ TRỊ THIẾU:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0])

    nulls.to_csv(os.path.join(OUTPUT_DIR, "missing_values.csv"))

    # ======================
    # 3. TOP ARTISTS
    # ======================
    if 'artist' in df.columns:
        top_artists = df['artist'].value_counts().head(10)

        print("\nTOP NGHỆ SĨ:")
        print(top_artists.to_string())

        top_artists.to_csv(os.path.join(OUTPUT_DIR, "top_artists.csv"))
    else:
        top_artists = None
        print("\nKhông có cột 'artist'")

    # ======================
    # 4. POPULARITY THEO DECADE
    # ======================
    if 'decade' in df.columns and 'popularity' in df.columns:
        decade_popularity = (
            df.groupby('decade')['popularity']
            .mean()
            .sort_index()
        )

        print("\nPOPULARITY THEO THẬP NIÊN:")
        print(decade_popularity.to_string())

        decade_popularity.to_csv(os.path.join(OUTPUT_DIR, "decade_popularity.csv"))
    else:
        decade_popularity = None
        print("\nThiếu 'decade' hoặc 'popularity'")

    # ======================
    # 5. THỐNG KÊ MÔ TẢ
    # ======================
    print("\nTHỐNG KÊ MÔ TẢ:")
    stats = df[numeric_cols].describe()

    print(stats.to_string())
    stats.to_csv(os.path.join(OUTPUT_DIR, "eda_stats.csv"))

    # ======================
    # 6. CORRELATION (QUAN TRỌNG)
    # ======================
    if len(numeric_cols) > 0:
        corr = df[numeric_cols].corr()

        print("\nMA TRẬN TƯƠNG QUAN:")
        print(corr.to_string())

        corr.to_csv(os.path.join(OUTPUT_DIR, "correlation.csv"))

        # 🔥 Tìm feature liên quan nhất đến popularity
        if 'popularity' in corr.columns:
            pop_corr = corr['popularity'].sort_values(ascending=False)

            print("\nTƯƠNG QUAN VỚI POPULARITY:")
            print(pop_corr.to_string())

            pop_corr.to_csv(os.path.join(OUTPUT_DIR, "popularity_correlation.csv"))
    else:
        corr = None

    print("\nHOÀN THÀNH EDA")

    # ======================
    # RETURN đầy đủ để dùng tiếp
    # ======================
    return numeric_cols, corr, top_artists, decade_popularity