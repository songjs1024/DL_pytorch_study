import tensorflow as tf
from tensorflow import keras

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(output)
        return output, attention_weights

def positional_encoding(max_position, d_model):
    angle_rads = np.arange(max_position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def transformer_encoder(d_model, num_heads, num_layers, dff, input_vocab_size,
                       maximum_position_encoding, rate=0.1):
    inputs = tf.keras.layers.Input(shape=(None,))
    padding_mask = tf.keras.layers.Lambda(lambda x: tf.cast(tf.math.equal(x, 0), tf.float32))(inputs)
    seq_len = tf.shape(inputs)[1]
    position_encoding = positional_encoding(maximum_position_encoding, d_model)(inputs)

    x = tf.keras.layers.Embedding(input_vocab_size, d_model)(inputs)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x += position_encoding

    x = tf.keras.layers.Dropout(rate)(x)

    for _ in range(num_layers):
        x, _ = MultiHeadAttention(d_model, num_heads)(x, x, x, padding_mask)
        x = tf.keras.layers.Dropout(rate)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + x)

        x = tf.keras.layers.Dense(dff, activation='relu')(x)
        x = tf.keras.layers.Dense(d_model)(x)
        x = tf.keras.layers.Dropout(rate)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + x)

    outputs = tf.keras.layers.Dense(input_vocab_size)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# 하이퍼파라미터 설정
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = 10000
maximum_position_encoding = 10000

# 모델 생성
transformer = transformer_encoder(
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    dff=dff,
    input_vocab_size=input_vocab_size,
    maximum_position_encoding=maximum_position_encoding
)

# 모델 컴파일
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
transformer.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
