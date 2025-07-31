# ========== modeling/train.py ===========
# Caminho: charting_student_math_misunderstandings/modeling/train.py
import os

import joblib
import lightgbm as lgb
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedKFold

from charting_student_math_misunderstandings.config import (
    LGB_PARAMS,
    N_FOLDS,
    PROCESSED_DIR,
    RANDOM_STATE,
)


# utilitária MAP@3
def map3(y_true, y_pred, labels):
    order = np.argsort(-y_pred, axis=1)[:,:3]
    top3 = np.array(labels)[order]
    score = 0.0
    for i, true in enumerate(y_true):
        preds = top3[i]
        if true in preds:
            rank = list(preds).index(true) + 1
            score += 1.0 / rank
    return score / len(y_true)


def run():
    # 1) Carrega dados processados
    X_tr_text = sparse.load_npz(os.path.join(PROCESSED_DIR, 'X_text_train.npz'))
    X_te_text = sparse.load_npz(os.path.join(PROCESSED_DIR, 'X_text_test.npz'))
    X_tr_num  = np.load(os.path.join(PROCESSED_DIR, 'X_num_train.npy'))
    X_te_num  = np.load(os.path.join(PROCESSED_DIR, 'X_num_test.npy'))
    y_train   = joblib.load(os.path.join(PROCESSED_DIR, 'y_train.pkl'))

    # concatena features
    X_train = sparse.hstack([X_tr_text, X_tr_num])

    # label encoding
    labels = np.unique(y_train).tolist()
    y_idx  = np.array([labels.index(v) for v in y_train])

    # 2) Cross-validation
    oof_preds = np.zeros((X_train.shape[0], len(labels)))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_i, val_i) in enumerate(skf.split(X_train, y_idx)):
        dtrain = lgb.Dataset(X_train[tr_i], label=y_idx[tr_i])
        dval   = lgb.Dataset(X_train[val_i], label=y_idx[val_i])
        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            valid_sets=[dval],
            early_stopping_rounds=20,
            verbose_eval=50
        )
        oof_preds[val_i] = model.predict(X_train[val_i])
        # salva modelo
        model.save_model(os.path.join(PROCESSED_DIR, f"lgb_fold{fold}.txt"))

    # 3) Avalia CV
    print("CV MAP@3:", map3(y_train, oof_preds, labels))
    joblib.dump(oof_preds, os.path.join(PROCESSED_DIR, 'oof_preds.npy'))

    # 4) Treina full e salva
    full_train = lgb.Dataset(X_train, label=y_idx)
    final_model = lgb.train(LGB_PARAMS, full_train)
    final_model.save_model(os.path.join(PROCESSED_DIR, 'lgb_full.txt'))

if __name__ == "__main__":
    run()
