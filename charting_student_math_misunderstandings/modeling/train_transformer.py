# ========== train_transformer.py ===========
# Caminho: charting_student_math_misunderstandings/modeling/train_transformer.py
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from charting_student_math_misunderstandings.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    TRANSFORMER_CONFIG,
    RANDOM_STATE,
)


def compute_map3(eval_pred) -> Dict[str, float]:
    """
    Compute Mean Average Precision @ 3 for evaluation.
    """
    predictions, labels = eval_pred
    predictions = torch.softmax(torch.tensor(predictions), dim=-1)
    
    # Get top 3 predictions for each sample
    top3_indices = torch.topk(predictions, k=3, dim=-1).indices.numpy()
    
    map3_scores = []
    for i, label in enumerate(labels):
        if label in top3_indices[i]:
            # Find position of correct label in top 3
            position = np.where(top3_indices[i] == label)[0][0]
            # MAP@3 = 1 / (position + 1) if correct label is in top 3
            map3_scores.append(1.0 / (position + 1))
        else:
            map3_scores.append(0.0)
    
    return {"map3": np.mean(map3_scores)}


def load_and_prepare_data() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and prepare data for transformer training.
    """
    # Load prompts and labels
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_prompts.csv'))
    
    # Load label encoder
    import joblib
    label_encoder = joblib.load(os.path.join(PROCESSED_DIR, 'label_encoder.pkl'))
    
    # Encode labels
    train_df['label'] = label_encoder.transform(train_df['target'])
    
    # Split data
    train_data, eval_data = train_test_split(
        train_df, 
        test_size=0.2, 
        random_state=RANDOM_STATE,
        stratify=train_df['label']
    )
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_data)
    eval_dataset = Dataset.from_pandas(eval_data)
    
    # Create a dummy test dataset for prediction
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test_prompts.csv'))
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, eval_dataset, test_dataset


def tokenize_function(examples, tokenizer, max_length):
    """
    Tokenize the text data.
    """
    return tokenizer(
        examples['text_prompt'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


def setup_model_and_tokenizer(model_name: str, num_labels: int, use_peft: bool = False):
    """
    Setup model and tokenizer with optional PEFT configuration.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with GPU optimizations
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Enable gradient checkpointing for memory efficiency
    if TRANSFORMER_CONFIG.get('GPU_MEMORY_OPTIMIZATIONS', {}).get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled for memory efficiency")
    
    # Apply PEFT if requested
    if use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            **TRANSFORMER_CONFIG['PEFT_CONFIG']
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def train_transformer(
    exp_name: str,
    model_name: str = None,
    use_peft: bool = False,
    max_length: int = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
):
    """
    Train transformer model with specified parameters.
    """
    # Use config defaults if not specified
    model_name = model_name or TRANSFORMER_CONFIG['MODEL_NAME']
    max_length = max_length or TRANSFORMER_CONFIG['MAX_LEN']
    epochs = epochs or TRANSFORMER_CONFIG['EPOCHS']
    batch_size = batch_size or TRANSFORMER_CONFIG['BATCH_SIZE']
    learning_rate = learning_rate or TRANSFORMER_CONFIG['LEARNING_RATE']
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  No GPU detected, using CPU (training will be slow)")
    
    print(f"Training transformer model: {model_name}")
    print(f"Experiment: {exp_name}")
    print(f"PEFT: {use_peft}")
    print(f"Max length: {max_length}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Load data
    train_dataset, eval_dataset, test_dataset = load_and_prepare_data()
    
    # Load label encoder to get number of labels
    import joblib
    label_encoder = joblib.load(os.path.join(PROCESSED_DIR, 'label_encoder.pkl'))
    num_labels = len(label_encoder.classes_)
    
    print(f"Number of labels: {num_labels}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, num_labels, use_peft)
    
    # Tokenize datasets
    def tokenize_wrapper(examples):
        return tokenize_function(examples, tokenizer, max_length)
    
    train_dataset = train_dataset.map(tokenize_wrapper, batched=True)
    eval_dataset = eval_dataset.map(tokenize_wrapper, batched=True)
    
    # Setup output directory
    output_dir = os.path.join(MODELS_DIR, 'transformer', exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments with GPU optimizations
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=TRANSFORMER_CONFIG['WARMUP_STEPS'],
        weight_decay=TRANSFORMER_CONFIG['WEIGHT_DECAY'],
        gradient_accumulation_steps=TRANSFORMER_CONFIG['GRADIENT_ACCUMULATION_STEPS'],
        evaluation_strategy="steps",
        eval_steps=TRANSFORMER_CONFIG['EVAL_STEPS'],
        save_strategy="steps",
        save_steps=TRANSFORMER_CONFIG['SAVE_STEPS'],
        logging_steps=TRANSFORMER_CONFIG['LOGGING_STEPS'],
        load_best_model_at_end=True,
        metric_for_best_model="map3",
        greater_is_better=True,
        save_total_limit=3,
        # GPU optimizations
        fp16=TRANSFORMER_CONFIG.get('FP16', True),  # Mixed precision
        dataloader_pin_memory=TRANSFORMER_CONFIG.get('DATALOADER_PIN_MEMORY', False),
        remove_unused_columns=TRANSFORMER_CONFIG.get('GPU_MEMORY_OPTIMIZATIONS', {}).get('remove_unused_columns', True),
        dataloader_num_workers=TRANSFORMER_CONFIG.get('GPU_MEMORY_OPTIMIZATIONS', {}).get('dataloader_num_workers', 2),
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_map3,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save best model
    best_model_path = os.path.join(output_dir, 'best_checkpoint')
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    # Save training metrics
    metrics = trainer.evaluate()
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save label encoder
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
    
    print(f"Training completed. Model saved to: {best_model_path}")
    print(f"Best MAP@3: {metrics.get('eval_map3', 'N/A')}")
    
    return best_model_path


def main():
    parser = argparse.ArgumentParser(description='Train transformer model for math misconceptions')
    parser.add_argument('--exp', required=True, help='Experiment name (e.g., exp02)')
    parser.add_argument('--model_name', help='Model name (default from config)')
    parser.add_argument('--use_peft', action='store_true', help='Use Parameter Efficient Fine-Tuning')
    parser.add_argument('--max_length', type=int, help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    
    args = parser.parse_args()
    
    train_transformer(
        exp_name=args.exp,
        model_name=args.model_name,
        use_peft=args.use_peft,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
