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
    # 1. Histogram (2x2)
    # ======================
    important_cols = ['popularity', 'tempo', 'energy', 'danceability']
    cols = [col for col in important_cols if col in df.columns]

    if cols:
        plt.figure(figsize=(12, 8))

        for i, col in enumerate(cols):
            plt.subplot(2, 2, i + 1)
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(col)

        plt.suptitle("Phân phối các biến quan trọng", fontsize=16)
        plt.tight_layout()

        plt.savefig(os.path.join(OUTPUT_DIR, "hist_overview.png"))
        if show: plt.show()
        plt.close()

    # ======================
    # 2. Heatmap
    # ======================
    if corr is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")

        plt.savefig(os.path.join(OUTPUT_DIR, "heatmap.png"))
        if show: plt.show()
        plt.close()

    # ======================
    # 3. Scatter (2x2)
    # ======================
    if 'popularity' in df.columns:
        cols = [col for col in numeric_cols if col != 'popularity'][:4]

        if cols:
            plt.figure(figsize=(12, 8))

            for i, col in enumerate(cols):
                plt.subplot(2, 2, i + 1)
                sns.scatterplot(x=df[col], y=df['popularity'], alpha=0.5)
                plt.title(col)

            plt.suptitle("Quan hệ với Popularity", fontsize=16)
            plt.tight_layout()

            plt.savefig(os.path.join(OUTPUT_DIR, "scatter_overview.png"))
            if show: plt.show()
            plt.close()

    # ======================
    # 4. Top Artists
    # ======================
    if top_artists is not None:
        plt.figure(figsize=(10, 6))
        top_artists.sort_values(ascending=False).plot(kind='bar')
        plt.title("Top nghệ sĩ")
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "top_artists.png"))

        if show: plt.show()
        plt.close()

    # ======================
    # 5. Decade Trend
    # ======================
    if decade_popularity is not None:
        plt.figure(figsize=(10, 6))
        decade_popularity.sort_index().plot(marker='o')
        plt.title("Popularity theo thập niên")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "decade_popularity.png"))

        if show: plt.show()
        plt.close()

    # ======================
    # 6. Feature Importance
    # ======================
    if feature_importance is not None:
        plt.figure()
        feature_importance.sort_values().plot(kind='barh')
        plt.title("Feature Importance")

        plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
        if show: plt.show()
        plt.close()

    # ======================
    # 7. Model Evaluation (ĐÃ FIX)
    # ======================
    if y_test is not None and y_pred_rf is not None:
        print("DEBUG:", y_test is not None, y_pred is not None, y_pred_rf is not None)

        plt.figure(figsize=(12, 5))

        # Residual Plot
        plt.subplot(1, 2, 1)
        residuals = y_test - y_pred_rf
        sns.scatterplot(x=y_pred_rf, y=residuals, alpha=0.5)
        plt.axhline(y=0, linestyle='--')
        plt.title("Residual Plot")

        # Model Comparison (LUÔN VẼ)
        plt.subplot(1, 2, 2)

        # Linear Regression (nếu có)
        if y_pred is not None:
            sns.scatterplot(x=y_test, y=y_pred, label="LR", alpha=0.5)

        # Random Forest (luôn có)
        sns.scatterplot(x=y_test, y=y_pred_rf, label="RF", alpha=0.5)

        # Đường chuẩn y = x (model lý tưởng)
        min_val = min(y_test.min(), y_pred_rf.min())
        max_val = max(y_test.max(), y_pred_rf.max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

        plt.legend()
        plt.title("Model Comparison")

        plt.tight_layout()

        plt.savefig(os.path.join(OUTPUT_DIR, "model_evaluation.png"))
        if show: plt.show()
        plt.close()

    print("\nHOÀN THÀNH VẼ BIỂU ĐỒ")