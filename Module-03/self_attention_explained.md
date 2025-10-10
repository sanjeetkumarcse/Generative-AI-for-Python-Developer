
# Self-Attention Mechanism (Step-by-Step)

This document explains how **self-attention** works in transformer models with a simple numerical example.

---

## Step 1: Input

Input tensor `X` (shape: 1×3×4):

| Token | Values |
|--------|---------|
| 1 | [0.3745, 0.9507, 0.7319, 0.5987] |
| 2 | [0.1560, 0.1559, 0.0581, 0.8662] |
| 3 | [0.6011, 0.7081, 0.0206, 0.9699] |

---

## Step 2: Linear Projections

Random weight matrices `W_q`, `W_k`, and `W_v` are generated with shape (4×4).  
They project the input into **Query (Q)**, **Key (K)**, and **Value (V)** spaces.

---

## Step 3: Compute Q, K, V

Example values (rounded):

```
Q = [[-1.59, 0.16, 1.41, 0.20],
     [-0.65, -0.32, 0.83, 0.43],
     [-1.22, 0.05, 1.53, 0.22]]

K = [[0.42, 0.74, -0.76, 0.12],
     [-0.05, 0.60, -0.47, 0.84],
     [0.55, 0.89, -0.92, 0.15]]
```

---

## Step 4: Scaled Dot-Product

Compute similarity scores:

```
scores = Q × Kᵀ / √d_k
```

where `d_k = 4`. Example output:

```
scores =
[[0.10, 0.04, 0.12],
 [0.05, 0.02, 0.06],
 [0.09, 0.03, 0.11]]
```

---

## Step 5: Apply Softmax

Softmax converts scores to attention weights:

```
weights =
[[0.333, 0.331, 0.336],
 [0.334, 0.330, 0.336],
 [0.333, 0.331, 0.336]]
```

Each row sums to **1**.

---

## Step 6: Compute Output

```
output = weights × V
```

Example result:

```
output =
[[0.01, 0.08, -0.03, 0.11],
 [0.01, 0.08, -0.03, 0.11],
 [0.01, 0.08, -0.03, 0.11]]
```

---

## Summary Table

| Step | Operation | Shape |
|------|------------|--------|
| Input | X | (1, 3, 4) |
| Linear projections | Q, K, V | (1, 3, 4) |
| Dot product | scores = QKᵀ / √d_k | (1, 3, 3) |
| Softmax | attention_weights | (1, 3, 3) |
| Weighted sum | output = weights·V | (1, 3, 4) |

---

## Visualization

The following diagram illustrates the self-attention process:

![Self-Attention Diagram](A_diagram_illustrates_the_self-attention_mechanism.png)

---

**Key Takeaway:**  
Self-attention allows each token to attend to other tokens by computing similarity (via QKᵀ), scaling, normalizing (via softmax), and using these weights to produce a context-aware representation (output).
