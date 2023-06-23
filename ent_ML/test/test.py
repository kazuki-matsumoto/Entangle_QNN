import numpy as np

# データセットの準備
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # 入力データ
y = np.array([0, 1, 0, 1])  # 出力データ

# ハイパーパラメータの設定
batch_size = 2  # バッチサイズ
epochs = 3  # エポック数

# ミニバッチ学習の実行
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    indices = np.random.permutation(X.shape[0])  # データをシャッフルするためのインデックスを生成
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for i in range(0, X.shape[0], batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        # バッチごとの学習を実行する
        # ここでは、単純にバッチごとの平均値を計算して表示するだけの例とする
        batch_mean = np.mean(X_batch)
        print(f"Batch {i//batch_size + 1}: Mean = {batch_mean}")

    print()
