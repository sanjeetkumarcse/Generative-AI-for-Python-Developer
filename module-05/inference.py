import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# 1. Device setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# 2. Load tokenizer and fine-tuned model
# -----------------------------
MODEL_PATH = "./fine_tuned_model"  # folder of your saved model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)

# -----------------------------
# 3. Inference function
# -----------------------------
def generate_text(input_text, max_tokens=50):
    """
    Generates output from the fine-tuned multi-task model.
    Supports translation, summarization, and QA.
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# 4. Interactive prompt
# -----------------------------
if __name__ == "__main__":
    print("Multi-task Inference Script (Translation, Summarization, QA)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter your input: ").strip()
        if user_input.lower() == "exit":
            break
        output = generate_text(user_input)
        print(f"Output: {output}\n")
