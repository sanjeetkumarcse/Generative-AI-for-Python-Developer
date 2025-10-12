# Deep Learning Hyperparameter Cheat Sheet

## Network Architecture

| Hyperparameter              | Purpose                                                 | Typical Range / Values               | Example Use Case               |
| --------------------------- | ------------------------------------------------------- | ------------------------------------ | ------------------------------ |
| Number of layers            | Network depth; more layers → can learn complex patterns | 1–100+                               | Deep CNN for image recognition |
| Number of neurons per layer | Capacity of each layer to learn features                | 16–1024+                             | MLP for tabular data           |
| Activation function         | Introduces non-linearity                                | ReLU, Sigmoid, Tanh, LeakyReLU, GELU | All deep networks              |
| Dropout rate                | Prevent overfitting                                     | 0–0.5                                | CNN, MLP, RNN                  |

## Optimization

| Hyperparameter    | Purpose                                       | Typical Range / Values      | Example Use Case         |
| ----------------- | --------------------------------------------- | --------------------------- | ------------------------ |
| Optimizer         | Algorithm for weight updates                  | SGD, Adam, RMSprop, Adagrad | All deep learning models |
| Learning rate     | Step size for weight updates                  | 1e-5 – 1e-1                 | All deep networks        |
| Momentum          | Helps SGD converge faster, avoid local minima | 0.5–0.99                    | CNN, MLP                 |
| Beta1/Beta2       | Adam momentum terms                           | Beta1: 0.9, Beta2: 0.999    | Adam optimizer           |
| Weight decay / L2 | Penalizes large weights                       | 1e-5 – 1e-2                 | Prevent overfitting      |

## Training

| Hyperparameter          | Purpose                               | Typical Range / Values | Example Use Case |
| ----------------------- | ------------------------------------- | ---------------------- | ---------------- |
| Batch size              | Number of samples per gradient update | 16–1024                | CNN, MLP, RNN    |
| Epochs                  | Number of full dataset passes         | 10–500+                | All models       |
| Early stopping patience | Stop training if no improvement       | 5–20 epochs            | All models       |
| Gradient clipping       | Prevent exploding gradients           | 0.1–5                  | RNN, LSTM, GRU   |

## Regularization

| Hyperparameter       | Purpose                           | Typical Range / Values          | Example Use Case    |
| -------------------- | --------------------------------- | ------------------------------- | ------------------- |
| Dropout              | Randomly drops neurons            | 0.1–0.5                         | CNN, MLP, RNN       |
| Batch normalization  | Stabilizes and speeds up learning | momentum 0.9–0.99, epsilon 1e-5 | CNN, MLP            |
| L1/L2 regularization | Penalizes large weights           | 1e-5 – 1e-2                     | Prevent overfitting |

## CNN Specific

| Hyperparameter    | Purpose                           | Typical Range / Values | Example Use Case  |
| ----------------- | --------------------------------- | ---------------------- | ----------------- |
| Filter size       | Kernel dimensions for convolution | 3x3, 5x5, 7x7          | Image recognition |
| Number of filters | Feature maps per layer            | 16–512                 | CNN               |
| Stride            | Step size for sliding kernel      | 1–3                    | CNN               |
| Padding           | Preserve spatial dimensions       | 'same', 'valid'        | CNN               |

## RNN / LSTM / GRU

| Hyperparameter    | Purpose                            | Typical Range / Values | Example Use Case        |
| ----------------- | ---------------------------------- | ---------------------- | ----------------------- |
| Hidden state size | Number of units per layer          | 32–1024                | Sequence modeling, text |
| Sequence length   | Input sequence window              | Task-dependent         | Time series, NLP        |
| Dropout           | Regularization in recurrent layers | 0–0.5                  | LSTM, GRU               |

## Transformer / Attention

| Hyperparameter   | Purpose                          | Typical Range / Values | Example Use Case                  |
| ---------------- | -------------------------------- | ---------------------- | --------------------------------- |
| Number of heads  | Parallel attention heads         | 4–16                   | NLP: translation, text generation |
| Hidden size      | Embedding dimension              | 128–1024               | Transformer                       |
| Number of layers | Encoder/decoder depth            | 2–24                   | BERT, GPT                         |
| Feedforward size | Inner layer of transformer block | 2–4 × hidden size      | NLP tasks                         |
| Dropout          | Regularization                   | 0–0.5                  | Transformers                      |
