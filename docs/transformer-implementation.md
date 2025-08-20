# Transformer Fine-Tuning Implementation

This document describes the transformer fine-tuning implementation for the MAP Competition.

## Overview

The transformer implementation uses Hugging Face's Transformers library to fine-tune pre-trained language models for the math misconception classification task. The implementation includes:

- **Text Prompt Generation**: Structured prompts combining question text, multiple choice answers, and student explanations
- **Model Training**: Fine-tuning with Hugging Face Trainer, including PEFT support
- **MAP@3 Evaluation**: Custom evaluation metric for the competition
- **Prediction Pipeline**: Top-3 prediction generation for submission

## Architecture

### Files Structure

```
charting_student_math_misunderstandings/
├── config.py                    # Configuration including transformer settings
├── features.py                  # Feature generation including prompt creation
└── modeling/
    ├── train_transformer.py     # Training script with Hugging Face Trainer
    └── predict_transformer.py   # Prediction and submission generation
```

### Configuration

The transformer configuration is defined in `config.py`:

```python
TRANSFORMER_CONFIG = dict(
    MODEL_TYPE = "transformer",
    MODEL_NAME = "microsoft/DialoGPT-medium",  # Default model
    MAX_LEN = 256,
    EPOCHS = 3,
    BATCH_SIZE = 8,
    LEARNING_RATE = 2e-5,
    USE_PEFT = False,  # Parameter Efficient Fine-Tuning
    # ... additional settings
)
```

## Usage

### 1. Generate Features and Prompts

```bash
make features
```

This creates:
- `data/processed/train_prompts.csv` - Training prompts with labels
- `data/processed/test_prompts.csv` - Test prompts
- `data/processed/label_encoder.pkl` - Label encoder for target classes

### 2. Train Transformer Model

```bash
# Standard training
make train-transformer EXP=exp01

# With PEFT (Parameter Efficient Fine-Tuning)
make train-transformer-peft EXP=exp02

# With custom parameters
python charting_student_math_misunderstandings/modeling/train_transformer.py \
    --exp exp03 \
    --model_name "microsoft/DialoGPT-large" \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 1e-5
```

### 3. Generate Predictions

```bash
make predict-transformer EXP=exp01
```

This creates a submission file: `submission_transformer_exp01.csv`

## Model Artifacts

After training, the following artifacts are saved in `models/transformer/{exp_name}/`:

```
models/transformer/exp01/
├── best_checkpoint/           # Best model checkpoint
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── label_encoder.pkl
├── metrics.json              # Training metrics
└── training_args.bin         # Training arguments
```

## Key Features

### 1. Text Prompt Format

The `format_input()` function creates structured prompts:

```
Question: {question_text}
Multiple Choice Answer: {mc_answer}
Student Explanation: {student_explanation}

Based on the student's explanation, identify potential math misconceptions:
```

### 2. MAP@3 Evaluation

The `compute_map3()` function implements the competition's evaluation metric:

- Calculates Mean Average Precision at rank 3
- Returns 1/(position + 1) if correct label is in top 3
- Returns 0 if correct label is not in top 3

### 3. PEFT Support

Parameter Efficient Fine-Tuning (PEFT) is supported via LoRA:

```python
# Enable PEFT in config
TRANSFORMER_CONFIG['USE_PEFT'] = True

# Or use command line flag
--use_peft
```

### 4. Top-3 Prediction Format

Predictions are formatted as space-delimited strings:

```
Category1:Misconception1 Category2:Misconception2 Category3:Misconception3
```

## Supported Models

The implementation supports any Hugging Face model compatible with `AutoModelForSequenceClassification`. Recommended models:

- **DialoGPT**: Good for conversational/explanation text
- **BERT variants**: Robust general-purpose models
- **RoBERTa**: Improved BERT with better training
- **DeBERTa**: Enhanced BERT with disentangled attention

## Training Tips

### 1. Model Selection

- Start with smaller models for faster iteration
- Use larger models for better performance
- Consider domain-specific models if available

### 2. Hyperparameter Tuning

- **Learning Rate**: Start with 2e-5, try 1e-5 to 5e-5
- **Batch Size**: Adjust based on GPU memory
- **Max Length**: Balance between context and memory usage
- **Epochs**: Monitor validation MAP@3 to avoid overfitting

### 3. PEFT Usage

- Use PEFT for large models or limited GPU memory
- Adjust LoRA rank (`r`) and alpha for performance vs. efficiency
- Monitor trainable parameters percentage

### 4. Data Augmentation

Consider augmenting the training data:
- Synonym replacement
- Back-translation
- Prompt variations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable PEFT
   - Use smaller model

2. **Slow Training**
   - Use mixed precision training
   - Enable gradient accumulation
   - Use PEFT for large models

3. **Poor MAP@3 Scores**
   - Check label distribution
   - Try different learning rates
   - Increase model size
   - Add data augmentation

### Testing Setup

Run the test script to verify everything works:

```bash
make test-transformer
```

## Future Improvements

1. **Ensemble Methods**: Combine multiple transformer models
2. **Cross-Validation**: Implement k-fold cross-validation
3. **Advanced PEFT**: Try different PEFT methods (AdaLoRA, QLoRA)
4. **Prompt Engineering**: Experiment with different prompt formats
5. **Data Augmentation**: Implement more sophisticated augmentation techniques

## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [MAP@K Evaluation](https://en.wikipedia.org/wiki/Mean_average_precision)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
