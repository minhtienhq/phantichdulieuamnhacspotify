import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
import matplotlib.pyplot as plt

from config import MODEL_DIR, OUTPUT_DIR


def train_model(df, features):

    print("\nHUẤN LUYỆN MÔ HÌNH")

    df_model = df.dropna(subset=features + ['popularity'])

    X = df_model[features]
    y = df_model['popularity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ======================
    # KHAI BÁO MODEL
    # ======================
    pipeline_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    pipeline_rf = Pipeline([
        ('model', RandomForestRegressor(random_state=42))
    ])

    # ======================
    # TRAIN
    # ======================
    pipeline_lr.fit(X_train, y_train)
    pipeline_rf.fit(X_train, y_train)

    # ======================
    # PREDICT
    # ======================
    y_pred = pipeline_lr.predict(X_test)
    y_pred_rf = pipeline_rf.predict(X_test)

    # ======================
    # ĐÁNH GIÁ
    # ======================
    def evaluate(y_true, y_pred, name):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        print(f"\n{name}")
        print(f"R2: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")

        return r2, mse, mae

    r2_lr, mse_lr, mae_lr = evaluate(y_test, y_pred, "Linear Regression")
    r2_rf, mse_rf, mae_rf = evaluate(y_test, y_pred_rf, "Random Forest")

    # ======================
    # FEATURE IMPORTANCE (RF)
    # ======================
    importance = pipeline_rf.named_steps['model'].feature_importances_

    importance_df = pd.Series(
        importance,
        index=features
    ).sort_values(ascending=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"))

    print("Đã lưu file: feature_importance.csv")

    # ⚠️ KHÔNG gán vào df (tránh lỗi pipeline side-effect)
    # nếu muốn dùng thì return riêng

    # ======================
    # SO SÁNH MODEL (BIỂU ĐỒ)
    # ======================
    models = ['Linear Regression', 'Random Forest']
    r2_scores = [r2_lr, r2_rf]

    plt.figure()
    plt.bar(models, r2_scores)
    plt.title("So sánh R2 giữa các mô hình")
    plt.ylabel("R2 Score")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plt.show()
    plt.close()

    # ======================
    # LƯU KẾT QUẢ
    # ======================
    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted_LR": y_pred,
        "Predicted_RF": y_pred_rf
    })

    results.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    print("Đã lưu file: predictions.csv")

    # ======================
    # LƯU MODEL
    # ======================
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(pipeline_lr, os.path.join(MODEL_DIR, "pipeline_lr.pkl"))
    joblib.dump(pipeline_rf, os.path.join(MODEL_DIR, "pipeline_rf.pkl"))

    print("Đã lưu model vào thư mục models")

    return y_test, y_pred, y_pred_rf