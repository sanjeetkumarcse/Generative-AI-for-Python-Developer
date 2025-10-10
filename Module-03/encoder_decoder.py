# LLM Encoding, Attention Evolution, and Decoding Visualization using GPT-2

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load GPT-2 model with language head for decoding
model_name = "gpt2"
HF_API_KEY = ""
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)

# Input text
text = "Transformers are powerful models for NLP."
inputs = tokenizer(text, return_tensors="pt")

# Forward pass through model
with torch.no_grad():
    outputs = model(**inputs)

# Extract internal representations
embeddings = outputs.hidden_states[0].squeeze(0)
first_layer = outputs.hidden_states[1].squeeze(0)
last_layer = outputs.hidden_states[-1].squeeze(0)
attentions = outputs.attentions  # list of attention matrices per layer
logits = outputs.logits

# --- Step 1: Tokenization ---
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
print("Tokens:", tokens)

# --- Step 2: Visualize Embedding Evolution ---
plt.figure(figsize=(10,5))
sns.heatmap(embeddings.numpy(), cmap="Blues", cbar=False)
plt.title("Token Embeddings (Initial)")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Tokens")
plt.show()

plt.figure(figsize=(10,5))
sns.heatmap(last_layer.numpy(), cmap="Greens", cbar=False)
plt.title("Final Layer Representations")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Tokens")
plt.show()

# --- Step 3: Visualize Average Attention (Last Layer) ---
plt.figure(figsize=(6,6))
last_layer_attn = attentions[-1].squeeze(0)  # (heads, tokens, tokens)
sns.heatmap(last_layer_attn.mean(0).numpy(), xticklabels=tokens, yticklabels=tokens, cmap="mako")
plt.title("Average Attention (Last Layer)")
plt.show()

# --- Step 4: Layer-Wise Attention Visualization ---
num_layers = len(attentions)
num_heads = attentions[0].shape[1]

for layer_idx in range(num_layers):
    layer_attn = attentions[layer_idx].squeeze(0)
    fig, axes = plt.subplots(1, min(num_heads, 4), figsize=(15, 3))
    fig.suptitle(f"Attention Heads - Layer {layer_idx+1}", fontsize=14)
    for head_idx in range(min(num_heads, 4)):
        sns.heatmap(layer_attn[head_idx].numpy(), ax=axes[head_idx], cmap="coolwarm", cbar=False)
        axes[head_idx].set_title(f"Head {head_idx+1}")
        axes[head_idx].set_xticklabels(tokens, rotation=90)
        axes[head_idx].set_yticklabels(tokens, rotation=0)
    plt.show()

# --- Step 5: Compare Early vs Late Layer Attention Evolution ---
early_layer_attn = attentions[0].squeeze(0).mean(0)
late_layer_attn = attentions[-1].squeeze(0).mean(0)

fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.heatmap(early_layer_attn.numpy(), ax=axes[0], xticklabels=tokens, yticklabels=tokens, cmap="crest")
axes[0].set_title("Early Layer (Layer 1) Attention")
sns.heatmap(late_layer_attn.numpy(), ax=axes[1], xticklabels=tokens, yticklabels=tokens, cmap="rocket_r")
axes[1].set_title("Late Layer (Layer N) Attention")
plt.suptitle("Attention Evolution from Early to Late Layers", fontsize=14)
plt.show()

# --- Step 6: Decoding (Next Token Prediction) ---
next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
next_token = tokenizer.decode(next_token_id)
print(f"\nPredicted Next Token: {next_token}")

# --- Step 7: Text Generation (Decoding Continuation) ---
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=20, temperature=0.7, top_k=50, top_p=0.9)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("\nGenerated Text:\n", generated_text)

# --- Step 8: Print Shapes for Each Stage ---
print(f"\nEmbeddings shape: {embeddings.shape}")
print(f"First layer shape: {first_layer.shape}")
print(f"Last layer shape: {last_layer.shape}")
print(f"Attention matrix shape (last layer): {last_layer_attn.shape}")

# --- Step 9: Summary ---
print("\nStep Summary:")
print("1️⃣ Tokenization → Tokens converted to IDs")
print("2️⃣ Embedding → Each token mapped to a vector")
print("3️⃣ Self-Attention → Calculates token dependencies")
print("4️⃣ Multi-Head Attention Visualization → Each head captures different context")
print("5️⃣ Attention Evolution → Early layers focus on local context, late layers on global meaning")
print("6️⃣ Decoding → Predicts next token based on context")
print("7️⃣ Generation → Produces coherent text continuation")
