from transformers import T5TokenizerFast, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Load and prepare the dataset
df = pd.read_csv("/data/minkyu/P_project/sign_sentence_dataset_fixed.csv")
df["input_text"] = "correct: " + df["input_text"]  

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer and model
tokenizer = T5TokenizerFast.from_pretrained("paust/pko-t5-small")
model = T5ForConditionalGeneration.from_pretrained("paust/pko-t5-small")

# Preprocessing function
def preprocess(example):
    inputs = tokenizer(example["input_text"], max_length=64, truncation=True, padding="max_length")
    targets = tokenizer(example["target_text"], max_length=64, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess, batched=True)

# Train/validation split
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

# Training configuration
training_args = TrainingArguments(
    output_dir="./pko-t5-small-corrector",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Start training
trainer.train()
