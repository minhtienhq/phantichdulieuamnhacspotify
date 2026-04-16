import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import OUTPUT_DIR


def plot_all(df, corr, top_artists, decade_popularity,
             y_test=None, y_pred=None, y_pred_rf=None,
             show=True):

    print("\nVẼ BIỂU ĐỒ")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ======================
    # 1. Histogram
    # ======================
    if 'popularity' in df.columns:
        plt.figure()
        df['popularity'].hist(bins=30)
        plt.title("Phân phối Popularity")

        path = os.path.join(OUTPUT_DIR, "hist_popularity.png")
        plt.savefig(path)
        print("Đã lưu:", path)

        if show: plt.show()
        plt.close()

    # ======================
    # 2. Heatmap
    # ======================
    if corr is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')

        path = os.path.join(OUTPUT_DIR, "heatmap.png")
        plt.savefig(path)
        print("Đã lưu:", path)

        if show: plt.show()
        plt.close()

    # ======================
    # 3. Top Artists
    # ======================
    if top_artists is not None:
        plt.figure()
        top_artists.plot(kind='bar')
        plt.title("Top nghệ sĩ")

        path = os.path.join(OUTPUT_DIR, "top_artists.png")
        plt.savefig(path)
        print("Đã lưu:", path)

        if show: plt.show()
        plt.close()

    # ======================
    # 4. Decade Trend
    # ======================
    if decade_popularity is not None:
        plt.figure()
        decade_popularity.sort_index().plot(marker='o')
        plt.title("Popularity theo thập niên")

        path = os.path.join(OUTPUT_DIR, "decade_popularity.png")
        plt.savefig(path)
        print("Đã lưu:", path)

        if show: plt.show()
        plt.close()

    # ======================
    # 5. Scatter
    # ======================
    if 'danceability' in df.columns and 'popularity' in df.columns:
        plt.figure()
        plt.scatter(df['danceability'], df['popularity'])
        plt.title("Danceability vs Popularity")

        path = os.path.join(OUTPUT_DIR, "dance_vs_pop.png")
        plt.savefig(path)
        print("Đã lưu:", path)

        if show: plt.show()
        plt.close()

    # ======================
    # 6. Boxplot
    # ======================
    if 'popularity' in df.columns:
        plt.figure()
        sns.boxplot(x=df['popularity'])
        plt.title("Boxplot Popularity")

        path = os.path.join(OUTPUT_DIR, "boxplot_popularity.png")
        plt.savefig(path)
        print("Đã lưu:", path)

        if show: plt.show()
        plt.close()

    # ======================
    # 7. Feature Importance
    # ======================
    if hasattr(df, "feature_importance"):
        importance = df.feature_importance

        plt.figure()
        importance.plot(kind='bar')
        plt.title("Feature Importance (Random Forest)")

        path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        plt.savefig(path)
        print("Đã lưu:", path)

        if show: plt.show()
        plt.close()

    # ======================
    # 8. Residual Plot
    # ======================
    if y_test is not None and y_pred_rf is not None:
        residuals = y_test - y_pred_rf

        plt.figure()
        plt.scatter(y_pred_rf, residuals)
        plt.axhline(y=0)
        plt.title("Residual Plot (Random Forest)")

        path = os.path.join(OUTPUT_DIR, "residual_plot.png")
        plt.savefig(path)
        print("Đã lưu:", path)

        if show: plt.show()
        plt.close()

    # ======================
    # 9. LR vs RF
    # ======================
    if y_test is not None and y_pred is not None and y_pred_rf is not None:

        plt.figure()

        plt.plot(y_test.values, label="Actual")
        plt.plot(y_pred, label="Linear Regression")
        plt.plot(y_pred_rf, label="Random Forest")

        plt.legend()
        plt.title("So sánh dự đoán LR và RF")

        path = os.path.join(OUTPUT_DIR, "lr_vs_rf.png")
        plt.savefig(path)
        print("Đã lưu:", path)

        if show: plt.show()
        plt.close()

    print("\nHOÀN THÀNH VẼ BIỂU ĐỒ")