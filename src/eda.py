import os
import pandas as pd
from config import OUTPUT_DIR

def run_eda(df):
    print("\nPHÂN TÍCH DỮ LIỆU (EDA)")

    # ======================
    # FEATURES
    # ======================
    features = ['danceability', 'energy', 'tempo', 'valence', 'loudness']

    # Kiểm tra cột tồn tại
    features = [col for col in features if col in df.columns]

    print("Các đặc trưng sử dụng:", features)

    # ======================
    # TOP ARTISTS
    # ======================
    if 'artist' in df.columns:
        top_artists = df['artist'].value_counts().head(10)

        print("\nTOP NGHỆ SĨ:")
        print(top_artists.to_string())

        top_artists.to_csv(os.path.join(OUTPUT_DIR, "top_artists.csv"))
        print("Đã lưu file: top_artists.csv")
    else:
        top_artists = None
        print("Không có cột 'artist'")

    # ======================
    # POPULARITY THEO DECADE
    # ======================
    if 'decade' in df.columns and 'popularity' in df.columns:
        decade_popularity = (
            df.groupby('decade')['popularity']
            .mean()
            .sort_values(ascending=False)
        )

        print("\nMỨC ĐỘ PHỔ BIẾN THEO THẬP NIÊN:")
        print(decade_popularity.to_string())

        decade_popularity.to_csv(os.path.join(OUTPUT_DIR, "decade_popularity.csv"))
        print("Đã lưu file: decade_popularity.csv")
    else:
        decade_popularity = None
        print("Thiếu cột 'decade' hoặc 'popularity'")

    # ======================
    # DESCRIPTIVE STATS
    # ======================
    print("\nTHỐNG KÊ MÔ TẢ:")
    stats = df.describe()

    print(stats.to_string())

    stats.to_csv(os.path.join(OUTPUT_DIR, "eda_stats.csv"))
    print("Đã lưu file: eda_stats.csv")

    # ======================
    # CORRELATION
    # ======================
    if len(features) > 0:
        corr = df[features].corr()

        print("\nMA TRẬN TƯƠNG QUAN:")
        print(corr.to_string())

        corr.to_csv(os.path.join(OUTPUT_DIR, "correlation.csv"))
        print("Đã lưu file: correlation.csv")
    else:
        corr = None

    print("\nHOÀN THÀNH EDA")

    return features, top_artists, decade_popularity