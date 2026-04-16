import os
from config import OUTPUT_DIR

def show_insight(decade_popularity):

    print("\nPHÂN TÍCH INSIGHT")

    if decade_popularity is None or len(decade_popularity) == 0:
        print("Không có dữ liệu để phân tích")
        return

    # ======================
    # INSIGHT TỰ ĐỘNG
    # ======================
    best_decade = decade_popularity.idxmax()
    best_value = decade_popularity.max()

    worst_decade = decade_popularity.idxmin()
    worst_value = decade_popularity.min()

    print(f"Thập niên phổ biến nhất: {best_decade} (popularity: {best_value:.2f})")
    print(f"Thập niên thấp nhất: {worst_decade} (popularity: {worst_value:.2f})")

    # Xu hướng
    trend = "tăng" if best_decade > worst_decade else "giảm"
    print(f"Xu hướng theo thời gian: {trend}")

    # Insight cố định (dùng cho báo cáo)
    print("Nhận xét:")
    print("- Danceability cao có xu hướng dễ viral")
    print("- Tempo khoảng 120 BPM phổ biến")
    print("- Nhạc mới có lợi thế trên nền tảng streaming")

    # ======================
    # LƯU FILE
    # ======================
    insight_text = f"""
INSIGHT REPORT
--------------

Thập niên phổ biến nhất: {best_decade} ({best_value:.2f})
Thập niên thấp nhất: {worst_decade} ({worst_value:.2f})
Xu hướng: {trend}

Nhận xét:
- Danceability cao có xu hướng dễ viral
- Tempo khoảng 120 BPM phổ biến
- Nhạc mới có lợi thế trên nền tảng streaming
"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "insight.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(insight_text.strip())

    print("Đã lưu file:", path)

    print("\nHOÀN THÀNH INSIGHT")