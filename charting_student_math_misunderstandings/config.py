# ========== config.py ===========
# Caminho: charting_student_math_misunderstandings/config.py
import os

# diretórios
RAW_DIR      = os.path.join(os.getcwd(), 'data', 'raw')
INTERIM_DIR  = os.path.join(os.getcwd(), 'data', 'interim')
PROCESSED_DIR= os.path.join(os.getcwd(), 'data', 'processed')

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

# Cross-validation
N_FOLDS = 5
RANDOM_STATE = 42
