import pandas as pd
from datasets import Dataset, DatasetDict
import random


def load_and_prepare_dataset(csv_path, augment=True):
    df = pd.read_csv(csv_path)
    df = df[['Situation', 'emotion', 'empathetic_dialogues', 'labels']].dropna()

    # Apply data augmentation if enabled
    if augment:
        augmented_data = []
        for _, row in df.iterrows():
            # Keep original sample
            augmented_data.append(row)
            
            # Apply simple augmentation - shuffle words in the situation (with 30% probability)
            if random.random() < 0.3:
                words = row['Situation'].split()
                if len(words) > 3:  # Only shuffle if we have enough words
                    random.shuffle(words)
                    new_row = row.copy()
                    new_row['Situation'] = ' '.join(words)
                    augmented_data.append(new_row)
        
        # Convert back to DataFrame
        df = pd.DataFrame(augmented_data)







    df = df.sample(2000, random_state=42)
    df['input'] = df.apply(build_prompt, axis=1)
    df['output'] = df['labels']
    
    # Create a train-validation split (80-20)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Create dataset dictionary with splits
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df[['input', 'output']]),
        'validation': Dataset.from_pandas(val_df[['input', 'output']])
    })
    
    return dataset_dict

def build_prompt(row):
    return (
        f"The user feels {row['emotion']} because {row['Situation']}.\n"
        f"Customer said: {row['empathetic_dialogues']}"
    )