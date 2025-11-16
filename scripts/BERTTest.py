from pathlib import Path
import numpy as np

# הספרייה הראשית של הפרויקט
BASE_DIR = Path(__file__).resolve().parent.parent  # INFO_RETRIEVAL/
OUTPUT_FOLDER = BASE_DIR / "bert_vectors"           # שם תיקיית הווקטורים

MATRIX_FILE = OUTPUT_FOLDER / "bert_matrix.npy"
FILENAMES_FILE = OUTPUT_FOLDER / "bert_filenames.txt"

print("Looking for files in:", OUTPUT_FOLDER.resolve())

# טעינת המטריצה
bert_matrix = np.load(MATRIX_FILE)
print("Shape:", bert_matrix.shape)
print("Sample:", bert_matrix[0][:10])

# טעינת רשימת הקבצים
with open(FILENAMES_FILE, "r", encoding="utf-8") as f:
    filenames = f.read().splitlines()
print("Number of filenames:", len(filenames))
print("First 5 filenames:", filenames[:5])
