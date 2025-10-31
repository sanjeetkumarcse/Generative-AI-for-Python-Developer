import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset

# -----------------------------
# 1. Check device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# 2. Load pre-trained model & tokenizer
# -----------------------------
model_name = "google/flan-t5-small"  # Free, no API key
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# -----------------------------
# 3. Load dataset
# -----------------------------
dataset = load_dataset("json", data_files={"train": "data/train.json", "validation": "data/val.json"})

# -----------------------------
# 4. Tokenize dataset
# -----------------------------
def preprocess(example):
    input_text = example["input_text"]
    target_text = example["target_text"]
    model_inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(target_text, truncation=True, padding="max_length", max_length=32)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = dataset["train"].map(preprocess)
tokenized_val = dataset["validation"].map(preprocess)

# -----------------------------
# 5. Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=50,
    
    eval_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,  # Use mixed precision if GPU supports
    report_to="none"
)

# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val
)

# -----------------------------
# 7. Start fine-tuning
# -----------------------------
trainer.train()

# -----------------------------
# 8. Save model
# -----------------------------
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# -----------------------------
# 9. Test inference
# -----------------------------
test_input = "Translate this to French: Hello, how are you?"
inputs = tokenizer(test_input, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
