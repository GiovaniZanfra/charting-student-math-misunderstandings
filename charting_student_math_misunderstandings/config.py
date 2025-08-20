# ========== config.py ===========
# Caminho: charting_student_math_misunderstandings/config.py
import os

# diretórios
RAW_DIR      = os.path.join(os.getcwd(), 'data', 'raw')
INTERIM_DIR  = os.path.join(os.getcwd(), 'data', 'interim')
PROCESSED_DIR= os.path.join(os.getcwd(), 'data', 'processed')
MODELS_DIR   = os.path.join(os.getcwd(), 'models')

# TF-IDF
TFIDF_PARAMS = dict(
    max_features=20000,
    ngram_range=(1,2),
    strip_accents='unicode',
    min_df=3,
    max_df=0.9
)

# LightGBM
LGB_PARAMS = dict(
    objective='multiclass',
    learning_rate=0.1,
    num_leaves=31,
    n_estimators=200,
    verbose=-1
)

# Transformer Models
TRANSFORMER_CONFIG = dict(
    MODEL_TYPE = "transformer",
    MODEL_NAME = "microsoft/DialoGPT-medium",  # Default model, can be overridden
    MAX_LEN = 256,
    EPOCHS = 3,
    BATCH_SIZE = 8,
    LEARNING_RATE = 2e-5,
    WARMUP_STEPS = 500,
    WEIGHT_DECAY = 0.01,
    GRADIENT_ACCUMULATION_STEPS = 4,
    EVAL_STEPS = 100,
    SAVE_STEPS = 500,
    LOGGING_STEPS = 50,
    SEED = 42,
    USE_PEFT = False,  # Parameter Efficient Fine-Tuning
    PEFT_CONFIG = dict(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
)

# Cross-validation
N_FOLDS = 5
RANDOM_STATE = 42
