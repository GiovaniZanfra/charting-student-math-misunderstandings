# ========== process.py ==========
# Caminho: charting_student_math_misunderstandings/process.py
import os

import pandas as pd

from charting_student_math_misunderstandings.config import INTERIM_DIR, RAW_DIR


def run():
    os.makedirs(INTERIM_DIR, exist_ok=True)

    # 1) Carrega dados raw
    train = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(RAW_DIR, "test.csv"))

    # 2) (Opcional) limpeza de texto bruto:
    # train['StudentExplanation'] = train['StudentExplanation'].str.lower().str.strip()

    # 3) Salva para interim
    train.to_csv(os.path.join(INTERIM_DIR, "train_interim.csv"), index=False)
    test.to_csv(os.path.join(INTERIM_DIR, "test_interim.csv"),  index=False)

if __name__ == "__main__":
    run()
