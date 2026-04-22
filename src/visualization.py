import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import OUTPUT_DIR
from sklearn.metrics import r2_score
import pandas as pd


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

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # ======================
    # 1. Histogram
    # ======================
    important_cols = ['popularity', 'tempo', 'energy', 'danceability']
    cols = [col for col in important_cols if col in df.columns]

    if cols:
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(cols):
            plt.subplot(2, 2, i + 1)
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(col)

        plt.suptitle("Phân phối các biến quan trọng")
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
    # 3. Scatter
    # ======================
    if 'popularity' in df.columns:
        cols = [col for col in numeric_cols if col != 'popularity'][:4]

        if cols:
            plt.figure(figsize=(12, 8))
            for i, col in enumerate(cols):
                plt.subplot(2, 2, i + 1)
                sns.scatterplot(x=df[col], y=df['popularity'], alpha=0.5)
                plt.title(col)

            plt.suptitle("Quan hệ với Popularity")
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
        plt.xticks(rotation=45)

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
    # 7. Model Evaluation
    # ======================
    if y_test is not None and y_pred_rf is not None:
        plt.figure(figsize=(12, 5))

        # Residual
        plt.subplot(1, 2, 1)
        residuals = y_test - y_pred_rf
        sns.scatterplot(x=y_pred_rf, y=residuals, alpha=0.5)
        plt.axhline(y=0, linestyle='--')
        plt.title("Residual Plot")

        # Comparison
        plt.subplot(1, 2, 2)

        if y_pred is not None:
            sns.scatterplot(x=y_test, y=y_pred, label="LR", alpha=0.5)

        sns.scatterplot(x=y_test, y=y_pred_rf, label="RF", alpha=0.5)

        min_val = min(y_test.min(), y_pred_rf.min())
        max_val = max(y_test.max(), y_pred_rf.max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

        plt.legend()
        plt.title("Model Comparison")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "model_evaluation.png"))

        if show: plt.show()
        plt.close()

    # ======================
    # 8. R2 COMPARISON (🔥 MỚI)
    # ======================
    if y_test is not None and y_pred_rf is not None:
        try:
            r2_rf = r2_score(y_test, y_pred_rf)
            r2_lr = r2_score(y_test, y_pred) if y_pred is not None else None

            if r2_lr is not None:
                r2_df = pd.DataFrame({
                    'Model': ['Linear Regression', 'Random Forest'],
                    'R2': [r2_lr, r2_rf]
                })
            else:
                r2_df = pd.DataFrame({
                    'Model': ['Random Forest'],
                    'R2': [r2_rf]
                })

            plt.figure(figsize=(6, 4))
            sns.barplot(x='Model', y='R2', data=r2_df)

            # Hiện số trên cột
            for i, v in enumerate(r2_df['R2']):
                plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

            plt.title("So sánh R2 giữa các mô hình")
            plt.ylabel("R2 Score")

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))

            if show: plt.show()
            plt.close()

        except Exception as e:
            print("Lỗi tính R2:", e)

    print("\nHOÀN THÀNH VẼ BIỂU ĐỒ")