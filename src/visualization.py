import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import OUTPUT_DIR


def plot_all(df,
             corr=None,
             top_artists=None,
             decade_popularity=None,
             y_test=None,
             y_pred=None,
             y_pred_rf=None,
             feature_importance=None,
             show=True):

    print("\nVẼ BIỂU ĐỒ")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ======================
    # 0. Lấy tất cả cột số
    # ======================
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # ======================
    # 1. Histogram
    # ======================
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Phân phối {col}")

        path = os.path.join(OUTPUT_DIR, f"hist_{col}.png")
        plt.savefig(path)

        if show: plt.show()
        plt.close()

    # ======================
    # 2. Heatmap
    # ======================
    if corr is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")

        path = os.path.join(OUTPUT_DIR, "heatmap.png")
        plt.savefig(path)

        if show: plt.show()
        plt.close()

    # ======================
    # 3. Scatter vs Popularity
    # ======================
    if 'popularity' in df.columns:
        for col in numeric_cols:
            if col != 'popularity':
                plt.figure()
                sns.scatterplot(x=df[col], y=df['popularity'], alpha=0.5)
                plt.title(f"{col} vs Popularity")

                path = os.path.join(OUTPUT_DIR, f"{col}_vs_popularity.png")
                plt.savefig(path)

                if show: plt.show()
                plt.close()

    # ======================
    # 4. Top Artists
    # ======================
    if top_artists is not None:
        plt.figure()
        top_artists.sort_values(ascending=False).plot(kind='bar')
        plt.title("Top nghệ sĩ")

        path = os.path.join(OUTPUT_DIR, "top_artists.png")
        plt.savefig(path)

        if show: plt.show()
        plt.close()

    # ======================
    # 5. Decade Trend
    # ======================
    if decade_popularity is not None:
        plt.figure()
        decade_popularity.sort_index().plot(marker='o')
        plt.title("Popularity theo thập niên")

        path = os.path.join(OUTPUT_DIR, "decade_popularity.png")
        plt.savefig(path)

        if show: plt.show()
        plt.close()

    # ======================
    # 6. Feature Importance
    # ======================
    if feature_importance is not None:
        plt.figure()
        feature_importance.sort_values().plot(kind='barh')
        plt.title("Feature Importance (Random Forest)")

        path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        plt.savefig(path)

        if show: plt.show()
        plt.close()

    # ======================
    # 7. Residual Plot
    # ======================
    if y_test is not None and y_pred_rf is not None:
        residuals = y_test - y_pred_rf

        plt.figure()
        sns.scatterplot(x=y_pred_rf, y=residuals, alpha=0.5)
        plt.axhline(y=0, linestyle='--')

        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot (Random Forest)")

        path = os.path.join(OUTPUT_DIR, "residual_plot.png")
        plt.savefig(path)

        if show: plt.show()
        plt.close()

    # ======================
    # 8. So sánh model
    # ======================
    if y_test is not None and y_pred is not None and y_pred_rf is not None:
        plt.figure()

        sns.scatterplot(x=y_test, y=y_pred, label="Linear Regression", alpha=0.5)
        sns.scatterplot(x=y_test, y=y_pred_rf, label="Random Forest", alpha=0.5)

        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.title("So sánh LR vs RF")

        path = os.path.join(OUTPUT_DIR, "lr_vs_rf.png")
        plt.savefig(path)

        if show: plt.show()
        plt.close()

    print("\nHOÀN THÀNH VẼ BIỂU ĐỒ")