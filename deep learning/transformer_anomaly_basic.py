import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np

# 트랜스포머 모델 정의
def create_transformer_model(input_shape, num_heads, num_layers, dff, rate=0.1):
    inputs = Input(shape=input_shape)
    x = inputs

    # Positional Encoding 추가
    positional_encoding = tf.keras.layers.Embedding(input_shape[0], input_shape[1])(tf.range(start=0, limit=input_shape[0], delta=1))
    x += positional_encoding

    for _ in range(num_layers):
        # Multi-Head Self-Attention
        x = MultiHeadAttention(num_heads=num_heads, key_dim=64)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x)

        # Feed Forward Neural Network
        x = tf.keras.layers.Dense(dff, activation='relu')(x)
        x = tf.keras.layers.Dense(input_shape[1])(x)
        x = LayerNormalization(epsilon=1e-6)(x)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # 출력층
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

# 이상탐지 데이터 생성 
seq_length = 50
num_samples = 1000
normal_data = np.random.randn(num_samples, seq_length, 1)
anomaly_data = np.random.randn(50, seq_length, 1) + 5  # 이상 데이터 생성

# 데이터 레이블 생성
normal_labels = np.zeros((num_samples, 1))
anomaly_labels = np.ones((50, 1))

# 데이터셋 생성
X = np.vstack([normal_data, anomaly_data])
y = np.vstack([normal_labels, anomaly_labels])

# 모델 생성
model = create_transformer_model((seq_length, 1), num_heads=2, num_layers=2, dff=32)

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(X, y, epochs=10, batch_size=64)

# 이상 탐지 예측
predictions = model.predict(X)
