# ========== features.py ===========
# Caminho: charting_student_math_misunderstandings/features.py
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from charting_student_math_misunderstandings.config import (
    INTERIM_DIR,
    PROCESSED_DIR,
    TFIDF_PARAMS,
)


def format_input(question_text, mc_answer, student_explanation):
    """
    Format input for transformer models with a structured prompt.
    """
    prompt = f"""Question: {question_text}
Multiple Choice Answer: {mc_answer}
Student Explanation: {student_explanation}

Based on the student's explanation, identify potential math misconceptions:"""
    return prompt


def build_and_save():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1) Carrega interim
    train = pd.read_csv(os.path.join(INTERIM_DIR, "train_interim.csv"))
    test  = pd.read_csv(os.path.join(INTERIM_DIR, "test_interim.csv"))

    # 2) Cria feature de texto combinado
    train['text_all'] = train['QuestionText'] + ' ' + train['MC_Answer'] + ' ' + train['StudentExplanation']
    test['text_all']  = test['QuestionText']  + ' ' + test['MC_Answer']  + ' ' + test['StudentExplanation']

    # 3) TF-IDF
    tfv = TfidfVectorizer(**TFIDF_PARAMS)
    X_tr = tfv.fit_transform(train['text_all'])
    X_te = tfv.transform(test['text_all'])
    joblib.dump(tfv, os.path.join(PROCESSED_DIR, 'tfidf.pkl'))

    # 4) Features numéricas
    tr_nums = np.vstack([
        train['StudentExplanation'].str.split().apply(len).values,
        train['StudentExplanation'].str.len().values,
        train['is_correct'].values
    ]).T
    te_nums = np.vstack([
        test['StudentExplanation'].str.split().apply(len).values,
        test['StudentExplanation'].str.len().values,
        test['is_correct'].values
    ]).T
    np.save(os.path.join(PROCESSED_DIR, 'X_num_train.npy'), tr_nums)
    np.save(os.path.join(PROCESSED_DIR, 'X_num_test.npy'), te_nums)

    # 5) Label encode target
    train['target'] = train['Category'] + ':' + train['Misconception']
    joblib.dump(train['target'].values, os.path.join(PROCESSED_DIR, 'y_train.pkl'))

    # 6) Salva sparse matrices (opcional: salve como .npz)
    from scipy import sparse
    sparse.save_npz(os.path.join(PROCESSED_DIR, 'X_text_train.npz'), X_tr)
    sparse.save_npz(os.path.join(PROCESSED_DIR, 'X_text_test.npz'),  X_te)

    # 7) Generate text prompts for transformer models
    print("Generating text prompts for transformer models...")
    
    # Create prompts for training data
    train['text_prompt'] = train.apply(
        lambda row: format_input(row['QuestionText'], row['MC_Answer'], row['StudentExplanation']), 
        axis=1
    )
    
    # Create prompts for test data
    test['text_prompt'] = test.apply(
        lambda row: format_input(row['QuestionText'], row['MC_Answer'], row['StudentExplanation']), 
        axis=1
    )
    
    # Save prompts
    train[['row_id', 'text_prompt', 'target']].to_csv(
        os.path.join(PROCESSED_DIR, 'train_prompts.csv'), index=False
    )
    test[['row_id', 'text_prompt']].to_csv(
        os.path.join(PROCESSED_DIR, 'test_prompts.csv'), index=False
    )
    
    # Create and save label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(train['target'].unique())
    joblib.dump(label_encoder, os.path.join(PROCESSED_DIR, 'label_encoder.pkl'))
    
    print(f"Saved {len(train)} training prompts and {len(test)} test prompts")
    print(f"Number of unique labels: {len(label_encoder.classes_)}")

if __name__ == "__main__":
    build_and_save()
