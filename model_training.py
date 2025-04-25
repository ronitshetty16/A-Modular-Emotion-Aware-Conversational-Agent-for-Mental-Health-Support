from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, TrainingArguments, Trainer
import torch
from transformers.trainer_callback import EarlyStoppingCallback


def tokenize_function(example, tokenizer):
    model_inputs = tokenizer(example['input'], padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(example['output'], padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model(tokenized_dataset, model_name="facebook/blenderbot-400M-distill"):
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    
    # Check if GPU has sufficient memory, otherwise default to CPU
    use_cuda = torch.cuda.is_available()
    device = "cpu"  # Default to CPU for training
    
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     eval_strategy="epoch",
    #     num_train_epochs=3,
    #     per_device_train_batch_size=1,  # Keep batch size small
    #     per_device_eval_batch_size=1,
    #     gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batch size
    #     save_strategy="epoch",
    #     logging_strategy="steps",
    #     logging_steps=10,
    #     fp16=False,  # Disable mixed precision since we're using CPU
    #     report_to="none",
    #     load_best_model_at_end=True,
    #     save_total_limit=2,
    #     # Force CPU training
    #     no_cuda=True,
    #     # Additional memory optimizations
    #     optim="adamw_torch",  # Use the torch implementation of AdamW
    # )
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",  # Changed from "epoch" to "steps"
        eval_steps=50,               # Evaluate every 50 steps
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        save_strategy="steps",       # Save checkpoints by steps
        save_steps=600,               # Save every 50 steps
        logging_strategy="steps",
        logging_steps=10,
        fp16=False,
        report_to="none",
        load_best_model_at_end=True,  # Make sure we load the best model
        save_total_limit=2,
        no_cuda=True,
        optim="adamw_torch",
        learning_rate=2e-5,           # Reduced learning rate
        weight_decay=0.01,            # Added weight decay for regularization
        warmup_steps=100,             # Add warmup steps
        # Early stopping
        metric_for_best_model="eval_loss",
        greater_is_better=False,      # For loss, lower is better
    )

    # Add regularization techniques
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        # Add early stopping callback
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["validation"],
    #     tokenizer=tokenizer,
    # )

    # Get the best model
    best_model_checkpoint = trainer.state.best_model_checkpoint
    if best_model_checkpoint:
        print(f"Loading best model from {best_model_checkpoint}")
        model = BlenderbotForConditionalGeneration.from_pretrained(best_model_checkpoint).to(device)


    
    trainer.train()
    model.save_pretrained("empathetic_blenderbot")
    tokenizer.save_pretrained("empathetic_blenderbot")
    return model, tokenizer