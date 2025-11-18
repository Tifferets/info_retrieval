"""
step6_BERT.py
==============
This script generates semantic vector embeddings for each text file using SBERT
(Sentence-BERT: all-mpnet-base-v2 model).  
The result is:
  - A matrix of embeddings (bert_matrix.npy)
  - A list of matching filenames (bert_filenames.txt)

These vectors will later be used for similarity-based retrieval or clustering.
"""

from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ============================================================
#                 PATH CONFIGURATION (RELATIVE)
# ============================================================
# BASE_DIR: the root folder of the project (two levels up from this file)
# This approach ensures portability across computers (no hardcoded paths).
BASE_DIR = Path(__file__).resolve().parent.parent

# Folder containing the cleaned-and-lemmatized text files
INPUT_FOLDER = BASE_DIR / "parliament_data" / "extracted_text"

# Folder where the output BERT vectors will be saved
OUTPUT_FOLDER = BASE_DIR / "bert_vectors"
OUTPUT_FOLDER.mkdir(exist_ok=True)  # Create folder if missing

# ============================================================
#                  LOAD SBERT PRE-TRAINED MODEL
# ============================================================
# all-mpnet-base-v2 → currently one of the strongest SBERT models
# Good balance of speed and accuracy.
model = SentenceTransformer("all-mpnet-base-v2")

# ============================================================
#                 LOAD INPUT TEXT FILES
# ============================================================
# Searches for all .txt files inside INPUT_FOLDER
text_files = sorted(INPUT_FOLDER.glob("*.txt"))

if len(text_files) == 0:
    print("❌ No .txt files found in:", INPUT_FOLDER)
    exit()  # Stop execution (no point continuing)

print(f"📄 Found {len(text_files)} text files in {INPUT_FOLDER}")

# Lists to accumulate the outputs
embeddings_list = []
filenames_list = []

# ============================================================
#                 ENCODE DOCUMENTS USING SBERT
# ============================================================
# tqdm → progress bar useful for long-running tasks
for file in tqdm(text_files, desc="Encoding with SBERT"):

    # Read the entire file content
    with open(file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Skip empty documents (if cleaning created empty files)
    if len(text) == 0:
        continue

    # Convert the text into a semantic numeric vector (embedding)
    emb = model.encode(text)

    embeddings_list.append(emb)       # Save vector
    filenames_list.append(file.name)  # Save filename (order stays synced)

# ============================================================
#                      SAVE OUTPUT FILES
# ============================================================
# bert_matrix.npy → 2D numpy array (#docs × vector_dim)
np.save(OUTPUT_FOLDER / "bert_matrix.npy", np.array(embeddings_list))

# bert_filenames.txt → filename list in the same order as the matrix rows
np.savetxt(OUTPUT_FOLDER / "bert_filenames.txt", filenames_list, fmt='%s')

print("\n✅ SBERT vectors created successfully!")
print(f"📁 Saved embeddings to: {OUTPUT_FOLDER}")
