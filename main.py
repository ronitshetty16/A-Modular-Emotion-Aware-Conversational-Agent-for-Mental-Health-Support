from data_processing import load_and_prepare_dataset
from model_training import train_model, tokenize_function
from chatbot_interface import launch_gradio
from transformers import BlenderbotTokenizer

if __name__ == "__main__":
    # Load and preprocess data
    import os
    import pandas as pd
    
    # Make sure the path to your CSV file is correct
    dataset = load_and_prepare_dataset("/home/ronit/nlp/emotion-emotion_69k.csv",augment=True)
    
    # Initialize tokenizer
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    
    # Tokenize both train and validation splits
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True
    )
    
    # Train model
    model, tokenizer = train_model(tokenized_dataset)
    
    # Launch interface
    launch_gradio(model, tokenizer)