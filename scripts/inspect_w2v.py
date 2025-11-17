import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # INFO_RETRIEVAL/
W2V_FOLDER = BASE_DIR / "w2v_lemma_and_words"      # תתקני לשם התיקייה שלך אם אחר

# --- CHOOSE WHAT TO VIEW ---
MATRIX = W2V_FOLDER / "w2v_lemm_matrix.npy"        # או w2v_word_matrix.npy
FILENAMES = W2V_FOLDER / "w2v_lemm_filenames.txt"

print("Loading matrix from:", MATRIX)
mat = np.load(MATRIX)

print("\nMatrix shape:", mat.shape)
df = pd.DataFrame(mat)

print("\n===== Showing first 5 rows =====\n")
print(df.head())

print("\n===== Showing first filenames =====\n")
with open(FILENAMES, "r", encoding="utf-8") as f:
    names = f.read().splitlines()
print(names[:10])
