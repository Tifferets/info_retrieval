from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ===== PATHS (Relative) =====
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FOLDER = BASE_DIR / "clean_xml"
OUTPUT_FOLDER = BASE_DIR / "bert_vectors"
OUTPUT_FOLDER.mkdir(exist_ok=True)

# ===== LOAD SBERT MODEL =====
model = SentenceTransformer("all-mpnet-base-v2")

# ===== LOAD TEXT FILES =====
text_files = sorted(INPUT_FOLDER.glob("*.txt"))

if len(text_files) == 0:
    print("❌ No .txt files found in:", INPUT_FOLDER)
    exit()

print(f"📄 Found {len(text_files)} text files in {INPUT_FOLDER}")

embeddings_list = []
filenames_list = []

for file in tqdm(text_files, desc="Encoding with SBERT"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if len(text) == 0:
        continue

    emb = model.encode(text)
    embeddings_list.append(emb)
    filenames_list.append(file.name)

# ===== SAVE OUTPUT =====
np.save(OUTPUT_FOLDER / "bert_matrix.npy", np.array(embeddings_list))
np.savetxt(OUTPUT_FOLDER / "bert_filenames.txt", filenames_list, fmt='%s')

print("\n✅ SBERT vectors created successfully!")
print(f"📁 Saved embeddings to: {OUTPUT_FOLDER}")
