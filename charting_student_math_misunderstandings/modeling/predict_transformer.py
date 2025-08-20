# ========== predict_transformer.py ===========
# Caminho: charting_student_math_misunderstandings/modeling/predict_transformer.py
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from charting_student_math_misunderstandings.config import MODELS_DIR


def load_model_and_tokenizer(exp_name: str):
    """
    Load trained model and tokenizer from experiment directory.
    """
    model_path = os.path.join(MODELS_DIR, 'transformer', exp_name, 'best_checkpoint')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load label encoder
    import joblib
    label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Check if model uses PEFT
    if os.path.exists(os.path.join(model_path, 'adapter_config.json')):
        model = PeftModel.from_pretrained(model, model_path)
        print("Loaded PEFT model")
    
    model.eval()
    
    return model, tokenizer, label_encoder


def predict_top3(
    model, 
    tokenizer, 
    label_encoder, 
    text_prompts: List[str], 
    max_length: int = 256
) -> List[str]:
    """
    Generate top-3 predictions for given text prompts.
    """
    predictions = []
    
    for prompt in text_prompts:
        # Tokenize
        inputs = tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probs, k=3, dim=-1)
            
            # Convert to labels
            top3_labels = []
            for idx in top3_indices[0]:
                label = label_encoder.inverse_transform([idx.item()])[0]
                top3_labels.append(label)
            
            # Format as space-delimited string
            prediction_str = ' '.join(top3_labels)
            predictions.append(prediction_str)
    
    return predictions


def create_submission(
    exp_name: str,
    output_file: str = None,
    max_length: int = 256
) -> str:
    """
    Create submission file for the competition.
    """
    # Load model and tokenizer
    model, tokenizer, label_encoder = load_model_and_tokenizer(exp_name)
    
    # Load test data
    from charting_student_math_misunderstandings.config import PROCESSED_DIR
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test_prompts.csv'))
    
    print(f"Generating predictions for {len(test_df)} test samples...")
    
    # Generate predictions
    predictions = predict_top3(
        model, 
        tokenizer, 
        label_encoder, 
        test_df['text_prompt'].tolist(),
        max_length
    )
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'row_id': test_df['row_id'],
        'Category:Misconception': predictions
    })
    
    # Save submission
    if output_file is None:
        output_file = f"submission_transformer_{exp_name}.csv"
    
    submission_df.to_csv(output_file, index=False)
    
    print(f"Submission saved to: {output_file}")
    print(f"Sample predictions:")
    for i in range(min(5, len(submission_df))):
        print(f"  {submission_df.iloc[i]['row_id']}: {submission_df.iloc[i]['Category:Misconception']}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Generate predictions with trained transformer model')
    parser.add_argument('--exp', required=True, help='Experiment name (e.g., exp02)')
    parser.add_argument('--output', help='Output file name (default: submission_transformer_{exp}.csv)')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    create_submission(
        exp_name=args.exp,
        output_file=args.output,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()
