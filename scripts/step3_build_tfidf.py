"""
Step 4: Build TF-IDF Matrices (BM25/Okapi) - Pure NLTK Stopwords
==================================================================
This script:
- Loads cleaned + lemmatized text files
- Builds TF-IDF matrices (word-level + lemma-level)
- Applies BM25 weighting (better ranking for IR tasks)
- Calculates feature importance using IG + Chi²
- Saves all matrices, features, filenames, and Excel reports
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from scipy.sparse import save_npz
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# NLTK for stopwords
import nltk
from nltk.corpus import stopwords


# ---------------------------------------------
# CLASS: BM25 TRANSFORMER
# ---------------------------------------------
class BM25Transformer:
    """
    BM25/Okapi Transformer

    Why we use BM25?
    ----------------
    TF-IDF punishes long documents unfairly. BM25 solves this by:
    - Normalizing by document length
    - Capping high term-frequency values
    - Improving ranking for IR tasks
    """

    def __init__(self, k1=1.5, b=0.75):
        """
        Initialize BM25 parameters.

        k1:
            Controls saturation. High k1 = term frequency matters more.
        b:
            Controls length normalization. b=0.75 is standard.
        """
        self.k1 = k1
        self.b = b

    def fit_transform(self, tf_matrix, doc_lengths, avg_doc_length, idf_vector):
        """
        Apply BM25 transformation on a TF matrix.

        Steps:
        ------
        1. Compute length normalization factor
        2. Apply BM25 formula for each term
        3. Multiply by IDF

        Returns:
            BM25-weighted sparse matrix
        """
        bm25_matrix = tf_matrix.copy()

        for i in range(bm25_matrix.shape[0]):
            doc_len = doc_lengths[i]
            length_norm = 1 - self.b + self.b * (doc_len / avg_doc_length)

            row = bm25_matrix.getrow(i)
            row_data = row.data

            # BM25 core formula
            row_data = row_data * (self.k1 + 1) / (row_data + self.k1 * length_norm)

            # Multiply by IDF
            col_indices = row.indices
            row_data = row_data * idf_vector[col_indices]

            # Write back into matrix
            bm25_matrix.data[bm25_matrix.indptr[i]:bm25_matrix.indptr[i+1]] = row_data

        return bm25_matrix


# ---------------------------------------------
# FUNCTION: Ensure NLTK stopwords exist
# ---------------------------------------------
def download_nltk_data():
    """
    Download NLTK stopwords if they are not installed.

    Why?
    ----
    TF-IDF usually removes very common words.
    This project requires using ONLY the official NLTK list.
    """
    print("\n📥 Checking NLTK stopwords...")
    try:
        _ = stopwords.words('english')
        print("✅ NLTK stopwords already available")
    except LookupError:
        print("📥 Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
        print("✅ Download completed!")


# ---------------------------------------------
# FUNCTION: Load NLTK stopwords
# ---------------------------------------------
def get_nltk_stopwords():
    """
    Return the PURE NLTK English stopword list (without additions).
    This ensures academic reproducibility.
    """
    print("\n🛑 Loading NLTK stopwords...")
    download_nltk_data()
    nltk_stopwords = set(stopwords.words('english'))

    print(f"   • Loaded {len(nltk_stopwords)} stopwords")
    print("   • No custom words added")
    return nltk_stopwords


# ---------------------------------------------
# FUNCTION: Load documents from folder
# ---------------------------------------------
def load_documents(folder_path):
    """
    Loads every .txt file from a folder and returns:
    - list of documents
    - list of filenames

    Why?
    ----
    TF-IDF requires a list of raw strings.
    We also save filenames to re-identify documents later.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    print(f"\n📂 Loading documents from: {folder}")

    txt_files = sorted(list(folder.glob('*.txt')))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {folder}")

    documents = []
    filenames = []

    for txt_file in tqdm(txt_files, desc="Loading files"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
                documents.append(text)
                filenames.append(txt_file.stem)
        except Exception as e:
            print(f"⚠️ Error reading {txt_file.name}: {e}")
            documents.append("")
            filenames.append(txt_file.stem)

    # Remove empty docs
    valid_docs = [
        (doc, fname) for doc, fname in zip(documents, filenames)
        if doc.strip()
    ]

    documents = [doc for doc, _ in valid_docs]
    filenames = [fname for _, fname in valid_docs]

    print(f"✅ Loaded {len(documents)} valid documents")
    return documents, filenames


# ---------------------------------------------
# FUNCTION: Information Gain
# ---------------------------------------------
def calculate_information_gain(X, y, feature_names):
    """
    Compute Information Gain for each feature.

    Why IG?
    -------
    IG tells us:
      "How much does this word help differentiate documents?"

    We create pseudo-labels by binning document lengths
    because this is an *unsupervised* dataset.
    """
    print("\n📊 Calculating Information Gain...")

    doc_lengths = np.array(X.sum(axis=1)).flatten()
    n_bins = min(10, len(doc_lengths) // 10)

    pseudo_labels = pd.cut(doc_lengths, bins=n_bins, labels=False)

    mi_scores = mutual_info_classif(X, pseudo_labels, random_state=42)

    ig_df = pd.DataFrame({
        'feature': feature_names,
        'information_gain': mi_scores
    }).sort_values('information_gain', ascending=False).reset_index(drop=True)

    print(f"✅ IG computed for {len(feature_names)} features")
    return ig_df


# ---------------------------------------------
# FUNCTION: Chi-Squared
# ---------------------------------------------
def calculate_chi_squared(X, y, feature_names):
    """
    Compute Chi² for each word.

    Why Chi²?
    ---------
    Chi² finds words that appear in specific "bins" (pseudo-groups)
    more often than expected by chance.
    """
    print("\n📊 Calculating Chi-squared...")

    doc_lengths = np.array(X.sum(axis=1)).flatten()
    n_bins = min(10, len(doc_lengths) // 10)

    pseudo_labels = pd.cut(doc_lengths, bins=n_bins, labels=False)

    chi2_scores, p_values = chi2(X, pseudo_labels)

    chi2_df = pd.DataFrame({
        'feature': feature_names,
        'chi_squared': chi2_scores,
        'p_value': p_values
    }).sort_values('chi_squared', ascending=False).reset_index(drop=True)

    print(f"✅ Chi² computed for {len(feature_names)} features")
    return chi2_df


# ---------------------------------------------
# FUNCTION: Build TF-IDF (optionally with BM25)
# ---------------------------------------------
def build_tfidf_matrix(documents, filenames, matrix_name,
                       min_df=5, max_df=0.95, max_features=10000,
                       use_bm25=True, stopwords_set=None):
    """
    Build a TF-IDF matrix and optionally convert it to BM25.

    Why these settings?
    -------------------
    • min_df=5   → remove extremely rare noise
    • max_df=0.95 → remove extremely common words
    • max_features=10000 → limits dimensionality for speed
    • stopwords = NLTK only → consistent academic baseline
    """
    print(f"\n{'='*70}")
    print(f"🔨 Building {matrix_name}")
    print(f"{'='*70}")

    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words=list(stopwords_set),
        lowercase=True,
        token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 1),
        norm='l2',
        use_idf=True,
        smooth_idf=True,
    )

    print("\n🔄 Fitting TF-IDF vectorizer...")
    tfidf_matrix = vectorizer.fit_transform(tqdm(documents, desc="Vectorizing"))
    feature_names = vectorizer.get_feature_names_out()

    print(f"\n✅ TF-IDF created: shape={tfidf_matrix.shape}")

    # Apply BM25
    if use_bm25:
        print("\n🔄 Applying BM25 transformation...")
        doc_lengths = np.array(tfidf_matrix.sum(axis=1)).flatten()
        avg_doc_length = doc_lengths.mean()
        idf_vector = vectorizer.idf_

        bm25_matrix = BM25Transformer().fit_transform(
            tfidf_matrix, doc_lengths, avg_doc_length, idf_vector
        )
        final_matrix = bm25_matrix
        print("✅ BM25 applied")
    else:
        final_matrix = tfidf_matrix

    stats = {
        'matrix_name': matrix_name,
        'num_documents': len(documents),
        'num_features': len(feature_names),
        'sparsity': (1 - final_matrix.nnz / (final_matrix.shape[0] * final_matrix.shape[1])) * 100,
        'non_zero_elements': final_matrix.nnz,
        'use_bm25': use_bm25
    }

    return final_matrix, feature_names, vectorizer, stats


# ---------------------------------------------
# FUNCTION: Export top features to Excel
# ---------------------------------------------
def export_to_excel(word_ig, word_chi2, lemm_ig, lemm_chi2,
                    word_stats, lemm_stats, output_file):
    """
    Save all feature importance tables into an Excel file.

    Why Excel?
    ----------
    • Easy for analysis  
    • Allows lecturers/graders to inspect results  
    """
    print(f"\n📊 Exporting to Excel: {output_file}")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        word_ig.head(100).to_excel(writer, sheet_name='Word_IG', index=False)
        word_chi2.head(100).to_excel(writer, sheet_name='Word_Chi2', index=False)
        lemm_ig.head(100).to_excel(writer, sheet_name='Lemm_IG', index=False)
        lemm_chi2.head(100).to_excel(writer, sheet_name='Lemm_Chi2', index=False)

        stats_df = pd.DataFrame([word_stats, lemm_stats])
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

    print("✅ Excel created successfully!")


# ---------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------
def main():
    """
    Main execution pipeline.
    Loads files, builds matrices, saves outputs.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║    Step 4: Build TF-IDF Matrices (BM25)                     ║
║    Using PURE NLTK Stopwords (No Custom Additions)          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Load stopwords
    nltk_stopwords = get_nltk_stopwords()

    # Folder paths
    CLEAN_TEXT_FOLDER = r"C:\Users\USER\Desktop\school work\Year 5\aichzur meida\clean_xml"
    LEMMATIZED_FOLDER = r"C:\Users\USER\Desktop\school work\Year 5\aichzur meida\lemmatized_files"
    OUTPUT_FOLDER = input("Enter path for output folder: ").strip()

    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    MIN_DF = 5
    MAX_DF = 0.95
    MAX_FEATURES = 10000

    # Load files
    clean_docs, clean_filenames = load_documents(CLEAN_TEXT_FOLDER)
    lemm_docs, lemm_filenames = load_documents(LEMMATIZED_FOLDER)

    # TF-IDF (word)
    word_matrix, word_features, word_vectorizer, word_stats = build_tfidf_matrix(
        documents=clean_docs,
        filenames=clean_filenames,
        matrix_name="TFIDF-Word",
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        use_bm25=True,
        stopwords_set=nltk_stopwords
    )

    save_npz(Path(OUTPUT_FOLDER) / "tfidf_word_matrix.npz", word_matrix)

    # Save filenames + features
    with open(Path(OUTPUT_FOLDER) / "tfidf_word_filenames.txt", "w") as f:
        f.write("\n".join(clean_filenames))

    with open(Path(OUTPUT_FOLDER) / "tfidf_word_features.txt", "w") as f:
        f.write("\n".join(word_features))

    # TF-IDF (lemma)
    lemm_matrix, lemm_features, lemm_vectorizer, lemm_stats = build_tfidf_matrix(
        documents=lemm_docs,
        filenames=lemm_filenames,
        matrix_name="TFIDF-Lemm",
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        use_bm25=True,
        stopwords_set=nltk_stopwords
    )

    save_npz(Path(OUTPUT_FOLDER) / "tfidf_lemm_matrix.npz", lemm_matrix)

    with open(Path(OUTPUT_FOLDER) / "tfidf_lemm_filenames.txt", "w") as f:
        f.write("\n".join(lemm_filenames))

    with open(Path(OUTPUT_FOLDER) / "tfidf_lemm_features.txt", "w") as f:
        f.write("\n".join(lemm_features))

    # Feature importance
    word_ig = calculate_information_gain(word_matrix, None, word_features)
    word_chi2 = calculate_chi_squared(word_matrix, None, word_features)

    lemm_ig = calculate_information_gain(lemm_matrix, None, lemm_features)
    lemm_chi2 = calculate_chi_squared(lemm_matrix, None, lemm_features)

    # Export Excel
    excel_file = Path(OUTPUT_FOLDER) / "feature_importance.xlsx"
    export_to_excel(word_ig, word_chi2, lemm_ig, lemm_chi2,
                    word_stats, lemm_stats, excel_file)

    print("\n🎉 All TF-IDF + BM25 matrices successfully created!")


if __name__ == "__main__":
    main()
