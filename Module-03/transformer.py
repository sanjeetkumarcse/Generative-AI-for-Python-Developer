import numpy as np
from scipy.special import softmax

def self_attention(X):
    batch_size, seq_len, d_model = X.shape
    d_k = d_model  # for simplicity

    # Random weight matrices
    W_q = np.random.randn(d_model, d_k)
    W_k = np.random.randn(d_model, d_k)
    W_v = np.random.randn(d_model, d_k)

    # Compute Q, K, V
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # Scaled dot-product attention
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = attention_weights @ V
    return output, attention_weights

# Example input
np.random.seed(42)
X = np.random.rand(1, 3, 4)
output, weights = self_attention(X)

print("Output:\n", output)
print("\nAttention Weights:\n", weights)
