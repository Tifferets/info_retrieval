"""
Step 3: Build Word2Vec Vectors
======================================

Build document vectors using Word2Vec embeddings.
Creates two matrix groups:
- W2V-Word: Clean text without punctuation, numbers, dates
- W2V-Lemm: Lemmatized text without punctuation, numbers, dates, stop-words

Usage:
    python step3_build_w2v_glove.py
"""

import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Word2Vec models
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# import nltk
# nltk.download('punkt', quiet=True)
# from nltk.tokenize import word_tokenize


def clean_text_for_embedding(text, remove_stopwords=False):
    """
    Clean text for embedding (remove punctuation, numbers, dates).
    
    Args:
        text (str): Input text
        remove_stopwords (bool): Whether to remove stop words
        
    Returns:
        list: List of cleaned tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove dates (various formats)
    text = re.sub(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', '', text)
    text = re.sub(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', '', text)
    
    # Remove numbers (but keep numbers within words like "4th" becomes "th")
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remove special characters and punctuation (but keep hyphens within words)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove empty tokens
    tokens = [t for t in tokens if t.strip()]
    
    # Remove stop words if requested
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    
    return tokens


def load_documents(folder_path):
    """Load all text documents from a folder"""
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
                text = f.read()
                if text.strip():
                    documents.append(text)
                    filenames.append(txt_file.stem)
        except Exception as e:
            print(f"⚠️  Error reading {txt_file.name}: {e}")
    
    print(f"✅ Loaded {len(documents)} valid documents")
    return documents, filenames


def train_word2vec(corpus_tokens, output_file, vector_size=300, window=5, min_count=5):
    """
    Train Word2Vec model
    
    Args:
        corpus_tokens: List of token lists (sentences/documents)
        output_file: Path to save model
        vector_size: Dimension of word vectors
        window: Context window size
        min_count: Minimum word frequency
        
    Returns:
        Trained Word2Vec model
    """
    print(f"\n🔄 Training Word2Vec model...")
    print(f"   • Vector size: {vector_size}")
    print(f"   • Window: {window}")
    print(f"   • Min count: {min_count}")
    
    model = Word2Vec(
        sentences=corpus_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=10,
        seed=42,
        sg=1  # Skip-gram model
    )
    
    # Save model
    model.save(str(output_file))
    print(f"✅ Model trained and saved: {output_file}")
    
    return model


def documents_to_vectors(documents, model, vector_size=300):
    """
    Convert documents to vectors using pre-trained Word2Vec model.
    Uses mean pooling of word vectors.
    
    Args:
        documents: List of text documents
        model: Trained Word2Vec model
        vector_size: Dimension of vectors
        
    Returns:
        numpy array of shape (num_docs, vector_size)
    """
    print(f"\n🔄 Converting documents to vectors...")
    
    vectors = []
    
    for doc in tqdm(documents, desc="Vectorizing", unit="doc"):
        # Get tokens (already cleaned)
        tokens = doc.split()
        
        # Get vectors for tokens that exist in vocabulary
        doc_vectors = []
        for token in tokens:
            if token in model.wv:
                doc_vectors.append(model.wv[token])
        
        # If no tokens found, use zero vector
        if doc_vectors:
            # Mean pooling
            doc_vector = np.mean(doc_vectors, axis=0)
        else:
            doc_vector = np.zeros(vector_size)
        
        vectors.append(doc_vector)
    
    return np.array(vectors)


def build_w2v_matrices(clean_folder, lemm_folder, output_folder, 
                       vector_size=300, window=5, min_count=5):
    """
    Build both W2V-Word and W2V-Lemm matrices
    
    Args:
        clean_folder: Path to clean text files
        lemm_folder: Path to lemmatized files
        output_folder: Path for output files
        vector_size: Dimension of vectors
        window: Context window
        min_count: Minimum word frequency
    """
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # W2V-Word Matrix
    # ============================================
    
    print(f"\n{'='*70}")
    print(f"🎯 Building W2V-Word Matrix")
    print(f"{'='*70}")
    
    # Load clean documents
    clean_docs, clean_filenames = load_documents(clean_folder)
    
    # Clean text for embedding (no stop words removal)
    print("\n🧹 Cleaning text for embedding (remove punctuation, numbers, dates)...")
    clean_tokens = []
    for doc in tqdm(clean_docs, desc="Cleaning", unit="doc"):
        tokens = clean_text_for_embedding(doc, remove_stopwords=False)
        if tokens:
            clean_tokens.append(tokens)
    
    # Train Word2Vec on clean text
    w2v_word_model_file = output_path / "w2v_word_model.bin"
    w2v_word = train_word2vec(
        clean_tokens, 
        w2v_word_model_file,
        vector_size=vector_size,
        window=window,
        min_count=min_count
    )
    
    # Reconstruct documents from tokens
    clean_docs_cleaned = [' '.join(tokens) for tokens in clean_tokens]
    
    # Convert to vectors
    print("\n📊 Converting clean documents to vectors...")
    w2v_word_vectors = documents_to_vectors(clean_docs_cleaned, w2v_word, vector_size)
    
    # Save W2V-Word matrix
    w2v_word_matrix_file = output_path / "w2v_word_matrix.npy"
    np.save(w2v_word_matrix_file, w2v_word_vectors)
    print(f"💾 Saved: {w2v_word_matrix_file}")
    
    # Save filenames
    w2v_word_filenames_file = output_path / "w2v_word_filenames.txt"
    with open(w2v_word_filenames_file, 'w', encoding='utf-8') as f:
        for fname in clean_filenames:
            f.write(fname + '\n')
    print(f"💾 Saved: {w2v_word_filenames_file}")
    
    print(f"\n✅ W2V-Word Matrix:")
    print(f"   • Shape: {w2v_word_vectors.shape}")
    print(f"   • Documents: {len(clean_filenames)}")
    print(f"   • Dimensions: {vector_size}")
    
    # ============================================
    # W2V-Lemm Matrix
    # ============================================
    
    print(f"\n{'='*70}")
    print(f"🎯 Building W2V-Lemm Matrix")
    print(f"{'='*70}")
    
    # Load lemmatized documents
    lemm_docs, lemm_filenames = load_documents(lemm_folder)
    
    # Clean text for embedding (with stop words removal)
    print("\n🧹 Cleaning lemmatized text (remove stop-words, punctuation, numbers, dates)...")
    lemm_tokens = []
    for doc in tqdm(lemm_docs, desc="Cleaning", unit="doc"):
        tokens = clean_text_for_embedding(doc, remove_stopwords=True)
        if tokens:
            lemm_tokens.append(tokens)
    
    # Train Word2Vec on lemmatized text
    w2v_lemm_model_file = output_path / "w2v_lemm_model.bin"
    w2v_lemm = train_word2vec(
        lemm_tokens,
        w2v_lemm_model_file,
        vector_size=vector_size,
        window=window,
        min_count=min_count
    )
    
    # Reconstruct documents from tokens
    lemm_docs_cleaned = [' '.join(tokens) for tokens in lemm_tokens]
    
    # Convert to vectors
    print("\n📊 Converting lemmatized documents to vectors...")
    w2v_lemm_vectors = documents_to_vectors(lemm_docs_cleaned, w2v_lemm, vector_size)
    
    # Save W2V-Lemm matrix
    w2v_lemm_matrix_file = output_path / "w2v_lemm_matrix.npy"
    np.save(w2v_lemm_matrix_file, w2v_lemm_vectors)
    print(f"💾 Saved: {w2v_lemm_matrix_file}")
    
    # Save filenames
    w2v_lemm_filenames_file = output_path / "w2v_lemm_filenames.txt"
    with open(w2v_lemm_filenames_file, 'w', encoding='utf-8') as f:
        for fname in lemm_filenames:
            f.write(fname + '\n')
    print(f"💾 Saved: {w2v_lemm_filenames_file}")
    
    print(f"\n✅ W2V-Lemm Matrix:")
    print(f"   • Shape: {w2v_lemm_vectors.shape}")
    print(f"   • Documents: {len(lemm_filenames)}")
    print(f"   • Dimensions: {vector_size}")
    
    # ============================================
    # Summary
    # ============================================
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETED!")
    print(f"{'='*70}")
    print(f"\n📊 Summary:")
    print(f"   W2V-Word: {w2v_word_vectors.shape}")
    print(f"   W2V-Lemm: {w2v_lemm_vectors.shape}")
    print(f"   Output folder: {output_path}")
    print(f"{'='*70}\n")
    
    return w2v_word_vectors, w2v_lemm_vectors, w2v_word, w2v_lemm


def main():
    """Main function"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║      Step 3: Build Word2Vec / GloVe Vectors                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Get paths from user
    print("\n📂 Configuration:\n")
    
    clean_folder = input("Path to clean text folder: ").strip()
    lemm_folder = input("Path to lemmatized text folder: ").strip()
    output_folder = input("Path for output folder: ").strip()
    
    # Parameters
    print("\n⚙️  Parameters (Enter for defaults):\n")
    
    vector_size_input = input("Vector size (default: 300): ").strip()
    vector_size = int(vector_size_input) if vector_size_input else 300
    
    window_input = input("Context window (default: 5): ").strip()
    window = int(window_input) if window_input else 5
    
    min_count_input = input("Minimum word frequency (default: 5): ").strip()
    min_count = int(min_count_input) if min_count_input else 5
    
    # Confirm
    print(f"\n{'='*70}")
    print(f"📋 Summary:")
    print(f"{'='*70}")
    print(f"Clean folder:    {clean_folder}")
    print(f"Lemm folder:     {lemm_folder}")
    print(f"Output folder:   {output_folder}")
    print(f"Vector size:     {vector_size}")
    print(f"Window:          {window}")
    print(f"Min count:       {min_count}")
    print(f"{'='*70}")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Cancelled")
        return
    
    # Build matrices
    build_w2v_matrices(
        clean_folder, lemm_folder, output_folder,
        vector_size=vector_size, window=window, min_count=min_count
    )


if __name__ == "__main__":
    main()