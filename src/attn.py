import tensorflow as tf
import numpy as np

class TokenVocab(tf.keras.layers.Layer):
    def __init__(self, e_size=128, vocab_size=60000, name=None):
        super(TokenVocab, self).__init__(name=name)
        self.e_size = e_size
        self.vocab_size = vocab_size
        self.vocab = self.add_weight("vocab", shape=[self.vocab_size, 1, self.e_size],
                                     initializer='random_normal')

    def call(self, x, logits=False):
        if logits:
            return tf.einsum("...l,Vl->...V", x, tf.squeeze(self.vocab, axis=1))
        else:
            return tf.gather(self.vocab, x, axis=0)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_size, value_size=None, model_size=None,
                 use_bias_p=False, use_softmax=False, use_non_lin_mix=False, 
                 sum_normalization=False, name=None):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.use_bias_p = use_bias_p
        self.use_softmax = use_softmax
        self.use_non_lin_mix = use_non_lin_mix
        self.sum_normalization = sum_normalization

    def build(self, input_shape):
        self.query_dense = tf.keras.layers.Dense(self.num_heads * self.key_size, use_bias=self.use_bias_p)
        self.key_dense = tf.keras.layers.Dense(self.num_heads * self.key_size, use_bias=self.use_bias_p)
        self.value_dense = tf.keras.layers.Dense(self.num_heads * self.value_size, use_bias=self.use_bias_p)
        self.final_projection = tf.keras.layers.Dense(self.model_size, use_bias=self.use_bias_p)

    def call(self, query, key, value, mask=None):
        query_heads = self._split_heads(self.query_dense(query))
        key_heads = self._split_heads(self.key_dense(key))
        value_heads = self._split_heads(self.value_dense(value))

        if self.sum_normalization:
            query_heads = query_heads / (tf.reduce_sum(query_heads, axis=-1, keepdims=True) + 1e-6)
            key_heads = key_heads / (tf.reduce_sum(key_heads, axis=-1, keepdims=True) + 1e-6)

        attn_logits = tf.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        if mask is not None:
            attn_logits += (mask * -1e9)

        if self.use_softmax:
            attn_weights = tf.nn.softmax(attn_logits / tf.sqrt(float(self.key_size)))
        elif self.use_non_lin_mix:
            y = tf.keras.layers.Dense(1, use_bias=False)(tf.constant([[1.0]]))
            attn_weights = tf.nn.softmax(attn_logits / tf.sqrt(float(self.key_size))) * tf.sigmoid(y * 10) + \
                           attn_logits * (1 - tf.sigmoid(y * 10))
        else:
            attn_weights = attn_logits

        attn = tf.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = self._combine_heads(attn)
        attn = self.final_projection(attn)
        return attn, attn_weights

    def _split_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        depth = self.num_heads * self.key_size
        reshaped = tf.reshape(x, (batch_size, seq_len, self.num_heads, depth // self.num_heads))
        return tf.transpose(reshaped, perm=[0, 2, 1, 3])

    def _combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[2]
        depth = self.num_heads * self.key_size
        reshaped = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(reshaped, (batch_size, seq_len, depth))

class MLP(tf.keras.layers.Layer):
    def __init__(self, widening_factor=4, second_layer=False, use_bias_p=False, outputdim=0, name=None):
        super(MLP, self).__init__(name=name)
        self.widening_factor = widening_factor
        self.second_layer = second_layer
        self.use_bias_p = use_bias_p
        self.outputdim = outputdim

    def build(self, input_shape):
        hiddens = input_shape[-1]
        self.dense1 = tf.keras.layers.Dense(self.widening_factor * hiddens, use_bias=self.use_bias_p)
        self.dense2 = tf.keras.layers.Dense(self.widening_factor * hiddens, use_bias=self.use_bias_p) if self.second_layer else None
        self.output_layer = tf.keras.layers.Dense(hiddens if self.outputdim == 0 else self.outputdim, use_bias=self.use_bias_p)

    def call(self, x):
        x = self.dense1(x)
        x = tf.nn.gelu(x)
        if self.second_layer:
            x = self.dense2(x)
            x = tf.nn.gelu(x)
        return self.output_layer(x)

class LNorm(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(LNorm, self).__init__(name=name)

    def call(self, x):
        return tf.keras.layers.LayerNormalization(axis=-1, scale=True, center=True)(x)

def layer_norm(x, name=None):
    return tf.keras.layers.LayerNormalization(axis=-1, scale=True, center=True)(x)

def create_pos_encoding(context_size, input_size, flip=False):
    pe = np.zeros((context_size, input_size))
    position = np.arange(0, context_size, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, input_size, 2) * (-np.log(10000.0) / input_size))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = pe[None]
    if flip:
        return tf.convert_to_tensor(np.flip(pe.squeeze(0), 0))
    else:
        return tf.convert_to_tensor(pe.squeeze(0))

def create_pos_encoding_diff(context_size, input_size):
    pe = np.zeros((context_size, input_size))
    position = np.arange(0, context_size, dtype=np.float32)[:, None]
    twoi = np.arange(0, input_size, 2)
    pe[:, 0::2] = np.sin(position / (10000**(twoi / input_size)))
    pe[:, 1::2] = np.cos(position / (10000**(twoi / input_size)))
    pe = pe[None]
    return tf.convert_to_tensor(pe.squeeze(0))
