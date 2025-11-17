"""
Step 5: Build SimCSE Vectors (Simple Version)
==============================================

Generate document embeddings using SimCSE.
This script only has ONE job:
→ Encode every document into a dense vector.

Why SimCSE?
-----------
SimCSE is a strong sentence-embedding model that produces 
high-quality semantic vectors. Unlike TF-IDF, it captures meaning, 
not just word frequency.

Usage:
    python step5_build_simcse_simple.py
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CLASS: SimCSE Encoder
# ============================================================
class SimCSEEncoder:
    """
    SimCSE encoder for generating document embeddings.

    Why we use this:
    ----------------
    • SimCSE is trained to bring similar sentences closer
    • Produces semantic vectors much better than averaging word vectors
    • The CLS token represents the whole sentence/document meaning
    """

    def __init__(self, model_name='princeton-nlp/sup-simcse-bert-base-uncased'):
        """Load SimCSE tokenizer + model"""

        print(f"\n🔄 Loading SimCSE model: {model_name}")
        print("   (First time: downloads ~400MB, may take 1–2 minutes...)")

        # Auto-select GPU if available (much faster)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {self.device}")

        # Load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()   # Disable training mode → faster, no dropout

        print(f"✅ SimCSE model loaded successfully!")

    def encode(self, texts, batch_size=8, max_length=512):
        """
        Convert a list of documents into embeddings.

        Why batching?
        -------------
        HuggingFace models cannot process hundreds of documents at once.
        Batch processing prevents memory errors.

        Why CLS token?
        --------------
        SimCSE is trained so that the CLS token embedding 
        carries the semantic meaning of the whole text.

        Inputs:
            texts       → list of strings
            batch_size  → # of documents processed at once
            max_length  → truncation limit to avoid GPU memory overflow

        Returns:
            NumPy array of shape (num_docs, 768)
        """
        all_embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        with torch.no_grad():   # Disable gradients → faster
            for i in tqdm(range(0, len(texts), batch_size),
                          desc="Encoding",
                          unit="batch",
                          total=num_batches):

                batch_texts = texts[i:i + batch_size]

                # Convert raw text into input IDs + attention masks
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,  # limit length
                    return_tensors='pt'
                ).to(self.device)

                # Forward pass through the model
                outputs = self.model(**inputs)

                # CLS embedding: index 0 represents entire document
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                all_embeddings.append(embeddings)

        # Stack all batches into a single matrix
        return np.vstack(all_embeddings)


# ============================================================
# FUNCTION: Load documents from folder
# ============================================================
def load_documents(folder_path):
    """
    Load all .txt files from the given folder.

    Why a separate loader?
    ----------------------
    • We need consistent order between vectors and filenames.
    • If a file is empty, we skip it to avoid zero-vector problems.

    Returns:
        documents → list of strings
        filenames → list of file stems
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"❌ Folder not found: {folder}")

    print(f"\n📂 Loading documents from: {folder}")

    txt_files = sorted(list(folder.glob('*.txt')))
    if not txt_files:
        raise FileNotFoundError(f"❌ No .txt files found in {folder}")

    print(f"📄 Found {len(txt_files)} files")

    documents = []
    filenames = []

    for txt_file in tqdm(txt_files, desc="Loading", unit="file"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()

                # Skip empty files
                if text:
                    documents.append(text)
                    filenames.append(txt_file.stem)

        except Exception as e:
            print(f"⚠️  Error reading {txt_file.name}: {e}")

    print(f"✅ Loaded {len(documents)} valid documents")
    return documents, filenames


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    """
    Main SimCSE pipeline:

    1. Ask user for input/output folders
    2. Load all .txt files
    3. Load SimCSE model
    4. Encode docs → 768-dimension vectors
    5. Save:
        • embeddings (.npy)
        • filenames (.txt)
        • run statistics
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║         Step 5: Build SimCSE Vectors                         ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # ---------------------------------------------------------
    # Default path suggestions
    # ---------------------------------------------------------
    print("📂 Configuration:\n")

    default_input = r"C:\Users\USER\Desktop\school work\Year 5\aichzur meida\parliament_data\extracted_text"
    default_output = r"C:\Users\USER\Desktop\school work\Year 5\aichzur meida\simcse_output"

    print(f"Default input:  {default_input}")
    input_folder = input("Input folder (Enter for default): ").strip() or default_input

    print(f"\nDefault output: {default_output}")
    output_folder = input("Output folder (Enter for default): ").strip() or default_output

    # ---------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------
    print("\n⚙️  Parameters (Enter for defaults):\n")

    batch_input = input("Batch size (default: 8): ").strip()
    batch_size = int(batch_input) if batch_input else 8

    max_len_input = input("Max token length (default: 512): ").strip()
    max_length = int(max_len_input) if max_len_input else 512

    # ---------------------------------------------------------
    # User confirmation
    # ---------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"📋 Summary:")
    print(f"{'='*70}")
    print(f"Input:      {input_folder}")
    print(f"Output:     {output_folder}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length} tokens")
    print(f"{'='*70}")

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Cancelled")
        return

    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Step 1: Load documents
    # ---------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"📖 LOADING DOCUMENTS")
    print(f"{'='*70}")
    documents, filenames = load_documents(input_folder)

    # ---------------------------------------------------------
    # Step 2: Load SimCSE Model
    # ---------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"🤖 INITIALIZING SIMCSE")
    print(f"{'='*70}")
    encoder = SimCSEEncoder()

    # ---------------------------------------------------------
    # Step 3: Generate embeddings
    # ---------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"🔄 GENERATING EMBEDDINGS")
    print(f"{'='*70}")
    print(f"Documents: {len(documents)}")
    print(f"Estimated time: CPU → 15–25 min | GPU → 5–10 min\n")

    embeddings = encoder.encode(documents, batch_size=batch_size, max_length=max_length)

    print(f"\n✅ Embeddings generated!")
    print(f"   Shape: {embeddings.shape}  (documents × 768 dims)")

    # ---------------------------------------------------------
    # Step 4: Save output files
    # ---------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"💾 SAVING RESULTS")
    print(f"{'='*70}")

    # Save embeddings matrix
    embeddings_file = output_path / "simcse_embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"✅ Saved: {embeddings_file}")

    # Save filenames
    filenames_file = output_path / "simcse_filenames.txt"
    with open(filenames_file, 'w', encoding='utf-8') as f:
        for fname in filenames:
            f.write(fname + '\n')
    print(f"✅ Saved: {filenames_file}")

    # Save run metadata
    stats_file = output_path / "simcse_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"num_documents: {len(documents)}\n")
        f.write(f"embedding_dim: {embeddings.shape[1]}\n")
        f.write(f"model: princeton-nlp/sup-simcse-bert-base-uncased\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"max_length: {max_length}\n")
    print(f"✅ Saved: {stats_file}")

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"✅ COMPLETED!")
    print(f"{'='*70}")
    print(f"📊 Summary:")
    print(f"   Documents:   {len(documents)}")
    print(f"   Dimensions:  {embeddings.shape[1]}")
    print(f"   Output:      {output_path}")
    print(f"{'='*70}\n")
    print(f"🎉 SimCSE vectors created successfully!")
    print(f"💡 Next: Build SBERT vectors")


if __name__ == "__main__":
    main()
