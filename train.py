from config import Config
from transformers import TrainingArguments, Trainer

def preprocess_function(examples, tokenizer):
    inputs = [f"{prompt}\n{completion}" for prompt, completion in zip(examples["prompt"], examples["completion"])]
    tokenized = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=Config.MAX_LENGTH
    )
    tokenized["labels"] = tokenized["input_ids"]

    return tokenized

def train(model, tokenizer, dataset):
    print(" --- Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True
    )

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        fp16=True,
        logging_steps=1,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(Config.OUTPUT_DIR)