# ========== features.py ===========
# Caminho: charting_student_math_misunderstandings/features.py
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from charting_student_math_misunderstandings.config import (
    INTERIM_DIR,
    PROCESSED_DIR,
    TFIDF_PARAMS,
)


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

if __name__ == "__main__":
    build_and_save()
