"""
Step 4: Build TF-IDF Matrices (BM25/Okapi)
===========================================

This script builds TF-IDF matrices using BM25/Okapi variant for:
1. Clean text files (TFIDF-Word)
2. Lemmatized text files (TFIDF-Lemm)

It also calculates feature importance using:
- Information Gain
- Chi-squared statistic

Author: Your Name
Date: 2025
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from scipy.sparse import save_npz, load_npz
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# For Information Gain calculation
from scipy.stats import entropy


class BM25Transformer:
    """
    BM25/Okapi Transformer
    
    BM25 is a ranking function used in information retrieval.
    It's an improvement over standard TF-IDF.
    
    Formula:
    BM25(q,d) = Σ IDF(qi) · (f(qi,d) · (k1 + 1)) / (f(qi,d) + k1 · (1 - b + b · |d|/avgdl))
    
    Where:
    - f(qi,d) = term frequency of qi in document d
    - |d| = length of document d
    - avgdl = average document length
    - k1 = term saturation parameter (usually 1.2-2.0)
    - b = length normalization parameter (usually 0.75)
    """
    
    def __init__(self, k1=1.5, b=0.75):
        """
        Initialize BM25 transformer
        
        Args:
            k1 (float): Controls term frequency saturation (default: 1.5)
            b (float): Controls document length normalization (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        
    def fit_transform(self, tf_matrix, doc_lengths, avg_doc_length, idf_vector):
        """
        Transform TF matrix to BM25 matrix
        
        Args:
            tf_matrix: Term frequency matrix (sparse)
            doc_lengths: Array of document lengths
            avg_doc_length: Average document length
            idf_vector: IDF values for each term
            
        Returns:
            BM25 matrix (sparse)
        """
        # Make a copy to avoid modifying original
        bm25_matrix = tf_matrix.copy()
        
        # Calculate BM25 for each document
        for i in range(bm25_matrix.shape[0]):
            # Get document length
            doc_len = doc_lengths[i]
            
            # Length normalization factor
            length_norm = 1 - self.b + self.b * (doc_len / avg_doc_length)
            
            # Get non-zero elements (terms that appear in this document)
            row = bm25_matrix.getrow(i)
            
            # Apply BM25 formula
            # BM25 = IDF(term) × (TF × (k1 + 1)) / (TF + k1 × length_norm)
            row_data = row.data
            row_data = row_data * (self.k1 + 1) / (row_data + self.k1 * length_norm)
            
            # Multiply by IDF
            # Get column indices for this row
            col_indices = row.indices
            row_data = row_data * idf_vector[col_indices]
            
            # Update the matrix
            bm25_matrix.data[bm25_matrix.indptr[i]:bm25_matrix.indptr[i+1]] = row_data
        
        return bm25_matrix


def load_documents(folder_path):
    """
    Load all text documents from a folder
    
    Args:
        folder_path (str): Path to folder containing text files
        
    Returns:
        tuple: (documents, filenames)
            - documents: list of text strings
            - filenames: list of file names (without extension)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    print(f"\n📂 Loading documents from: {folder}")
    
    # Find all text files
    txt_files = sorted(list(folder.glob('*.txt')))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {folder}")
    
    print(f"📄 Found {len(txt_files)} files")
    
    documents = []
    filenames = []
    
    # Load each file
    for txt_file in tqdm(txt_files, desc="Loading files", unit="file"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
                documents.append(text)
                filenames.append(txt_file.stem)  # filename without extension
        except Exception as e:
            print(f"⚠️  Error reading {txt_file.name}: {e}")
            # Add empty document to keep alignment
            documents.append("")
            filenames.append(txt_file.stem)
    
    # Remove empty documents
    valid_docs = [(doc, fname) for doc, fname in zip(documents, filenames) if doc.strip()]
    documents = [doc for doc, _ in valid_docs]
    filenames = [fname for _, fname in valid_docs]
    
    print(f"✅ Loaded {len(documents)} valid documents")
    
    return documents, filenames


def calculate_information_gain(X, y, feature_names):
    """
    Calculate Information Gain for each feature
    
    Information Gain measures how much information a feature gives us
    about the class. Higher IG = more informative feature.
    
    Formula:
    IG(Feature) = H(Class) - H(Class|Feature)
    
    Where H is entropy
    
    Args:
        X: Feature matrix (sparse or dense)
        y: Labels (for TF-IDF, we can use document IDs or create pseudo-labels)
        feature_names: List of feature names
        
    Returns:
        DataFrame with features and their IG scores
    """
    print("\n📊 Calculating Information Gain...")
    
    # For unsupervised case (no labels), we create pseudo-labels
    # based on document clusters or simply use mutual information
    
    # Use mutual_info_classif from sklearn
    # This works even without explicit labels
    # We'll create pseudo-labels based on document length or other heuristics
    
    # Simple approach: divide documents into groups based on their length
    doc_lengths = np.array(X.sum(axis=1)).flatten()
    n_bins = min(10, len(doc_lengths) // 10)  # Create bins
    pseudo_labels = pd.cut(doc_lengths, bins=n_bins, labels=False)
    
    # Calculate mutual information (similar to IG)
    mi_scores = mutual_info_classif(X, pseudo_labels, random_state=42)
    
    # Create DataFrame
    ig_df = pd.DataFrame({
        'feature': feature_names,
        'information_gain': mi_scores
    })
    
    # Sort by IG (descending)
    ig_df = ig_df.sort_values('information_gain', ascending=False)
    ig_df = ig_df.reset_index(drop=True)
    
    print(f"✅ Information Gain calculated for {len(feature_names)} features")
    
    return ig_df


def calculate_chi_squared(X, y, feature_names):
    """
    Calculate Chi-squared statistic for each feature
    
    Chi-squared test measures the independence between features and labels.
    Higher chi2 = feature is more dependent on the class.
    
    Args:
        X: Feature matrix (sparse or dense)
        y: Labels (pseudo-labels for unsupervised)
        feature_names: List of feature names
        
    Returns:
        DataFrame with features and their chi2 scores
    """
    print("\n📊 Calculating Chi-squared statistic...")
    
    # Create pseudo-labels (same as for IG)
    doc_lengths = np.array(X.sum(axis=1)).flatten()
    n_bins = min(10, len(doc_lengths) // 10)
    pseudo_labels = pd.cut(doc_lengths, bins=n_bins, labels=False)
    
    # Calculate chi-squared
    # Note: chi2 requires non-negative features
    chi2_scores, p_values = chi2(X, pseudo_labels)
    
    # Create DataFrame
    chi2_df = pd.DataFrame({
        'feature': feature_names,
        'chi_squared': chi2_scores,
        'p_value': p_values
    })
    
    # Sort by chi2 (descending)
    chi2_df = chi2_df.sort_values('chi_squared', ascending=False)
    chi2_df = chi2_df.reset_index(drop=True)
    
    print(f"✅ Chi-squared calculated for {len(feature_names)} features")
    
    return chi2_df


def build_tfidf_matrix(documents, filenames, matrix_name, 
                       min_df=5, max_df=0.95, max_features=10000,
                       use_bm25=True):
    """
    Build TF-IDF matrix with dimensionality reduction
    
    Args:
        documents: List of text documents
        filenames: List of document names
        matrix_name: Name for this matrix (e.g., "TFIDF-Word")
        min_df: Minimum document frequency (default: 5)
        max_df: Maximum document frequency (default: 0.95)
        max_features: Maximum number of features to keep (default: 10000)
        use_bm25: Whether to use BM25 instead of standard TF-IDF
        
    Returns:
        tuple: (matrix, feature_names, vectorizer, stats)
    """
    print(f"\n{'='*70}")
    print(f"🔨 Building {matrix_name}")
    print(f"{'='*70}")
    
    # Step 1: Create TF-IDF vectorizer
    print("\n⚙️  Creating TF-IDF vectorizer...")
    print(f"   • min_df (minimum document frequency): {min_df}")
    print(f"   • max_df (maximum document frequency): {max_df}")
    print(f"   • max_features: {max_features}")
    
    vectorizer = TfidfVectorizer(
        min_df=min_df,              # Ignore terms appearing in fewer than min_df documents
        max_df=max_df,              # Ignore terms appearing in more than max_df of documents
        max_features=max_features,  # Keep only top max_features by term frequency
        stop_words='english',       # Remove English stop words
        lowercase=True,             # Convert to lowercase
        token_pattern=r'(?u)\b\w+\b',  # Match words (alphanumeric sequences)
        ngram_range=(1, 1),         # Use only unigrams (single words)
        norm='l2',                  # L2 normalization
        use_idf=True,               # Use IDF weighting
        smooth_idf=True,            # Add 1 to document frequencies (smoothing)
        sublinear_tf=False          # Don't use sublinear TF scaling (we'll use BM25)
    )
    
    # Step 2: Fit and transform
    print("\n🔄 Fitting vectorizer and transforming documents...")
    tfidf_matrix = vectorizer.fit_transform(tqdm(documents, desc="Vectorizing"))
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\n✅ TF-IDF matrix created:")
    print(f"   • Shape: {tfidf_matrix.shape} (documents × features)")
    print(f"   • Sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")
    print(f"   • Non-zero elements: {tfidf_matrix.nnz:,}")
    
    # Step 3: Apply BM25 if requested
    if use_bm25:
        print("\n🔄 Applying BM25 transformation...")
        
        # Calculate document lengths
        doc_lengths = np.array(tfidf_matrix.sum(axis=1)).flatten()
        avg_doc_length = doc_lengths.mean()
        
        # Get IDF values from vectorizer
        idf_vector = vectorizer.idf_
        
        # Apply BM25
        bm25_transformer = BM25Transformer(k1=1.5, b=0.75)
        bm25_matrix = bm25_transformer.fit_transform(
            tfidf_matrix, doc_lengths, avg_doc_length, idf_vector
        )
        
        print(f"✅ BM25 applied")
        final_matrix = bm25_matrix
    else:
        final_matrix = tfidf_matrix
    
    # Step 4: Collect statistics
    stats = {
        'matrix_name': matrix_name,
        'num_documents': len(documents),
        'num_features': len(feature_names),
        'sparsity': (1 - final_matrix.nnz / (final_matrix.shape[0] * final_matrix.shape[1])) * 100,
        'non_zero_elements': final_matrix.nnz,
        'avg_doc_length': np.array(final_matrix.sum(axis=1)).flatten().mean(),
        'use_bm25': use_bm25
    }
    
    return final_matrix, feature_names, vectorizer, stats


def export_to_excel(word_ig, word_chi2, lemm_ig, lemm_chi2, 
                    word_stats, lemm_stats, output_file):
    """
    Export feature importance to Excel file
    
    Args:
        word_ig: Information Gain DataFrame for words
        word_chi2: Chi-squared DataFrame for words
        lemm_ig: Information Gain DataFrame for lemmas
        lemm_chi2: Chi-squared DataFrame for lemmas
        word_stats: Statistics for word matrix
        lemm_stats: Statistics for lemma matrix
        output_file: Path to output Excel file
    """
    print(f"\n📊 Exporting to Excel: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: TFIDF-Word - Information Gain
        word_ig.head(100).to_excel(
            writer, sheet_name='Word_InformationGain', index=False
        )
        
        # Sheet 2: TFIDF-Word - Chi-squared
        word_chi2.head(100).to_excel(
            writer, sheet_name='Word_ChiSquared', index=False
        )
        
        # Sheet 3: TFIDF-Lemm - Information Gain
        lemm_ig.head(100).to_excel(
            writer, sheet_name='Lemm_InformationGain', index=False
        )
        
        # Sheet 4: TFIDF-Lemm - Chi-squared
        lemm_chi2.head(100).to_excel(
            writer, sheet_name='Lemm_ChiSquared', index=False
        )
        
        # Sheet 5: Statistics Summary
        stats_df = pd.DataFrame([word_stats, lemm_stats])
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    print(f"✅ Excel file created: {output_file}")


def main():
    """
    Main function
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║         Step 4: Build TF-IDF Matrices (BM25)                 ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # ============================================
    # Configuration
    # ============================================
    
    # Folder paths
    CLEAN_TEXT_FOLDER = r"C:\Users\USER\Desktop\school work\Year 5\aichzur meida\clean_xml"
    LEMMATIZED_FOLDER = input("\nEnter path to lemmatized text folder: ").strip()
    OUTPUT_FOLDER = input("Enter path for output folder: ").strip()
    
    # Create output folder
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # Dimensionality reduction parameters
    MIN_DF = 5          # Minimum document frequency
    MAX_DF = 0.95       # Maximum document frequency (95% of documents)
    MAX_FEATURES = 10000  # Maximum number of features
    
    print(f"\n{'='*70}")
    print(f"📋 Configuration:")
    print(f"{'='*70}")
    print(f"Clean text folder:     {CLEAN_TEXT_FOLDER}")
    print(f"Lemmatized folder:     {LEMMATIZED_FOLDER}")
    print(f"Output folder:         {OUTPUT_FOLDER}")
    print(f"Min document freq:     {MIN_DF}")
    print(f"Max document freq:     {MAX_DF}")
    print(f"Max features:          {MAX_FEATURES}")
    print(f"{'='*70}")
    
    input("\nPress Enter to continue...")
    
    # ============================================
    # Load Documents
    # ============================================
    
    print("\n" + "="*70)
    print("📖 LOADING DOCUMENTS")
    print("="*70)
    
    # Load clean text (for TFIDF-Word)
    clean_docs, clean_filenames = load_documents(CLEAN_TEXT_FOLDER)
    
    # Load lemmatized text (for TFIDF-Lemm)
    lemm_docs, lemm_filenames = load_documents(LEMMATIZED_FOLDER)
    
    # ============================================
    # Build TFIDF-Word Matrix
    # ============================================
    
    word_matrix, word_features, word_vectorizer, word_stats = build_tfidf_matrix(
        documents=clean_docs,
        filenames=clean_filenames,
        matrix_name="TFIDF-Word",
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        use_bm25=True
    )
    
    # Save matrix
    word_matrix_file = Path(OUTPUT_FOLDER) / "tfidf_word_matrix.npz"
    save_npz(word_matrix_file, word_matrix)
    print(f"\n💾 Saved: {word_matrix_file}")
    
    # Save filenames mapping
    word_filenames_file = Path(OUTPUT_FOLDER) / "tfidf_word_filenames.txt"
    with open(word_filenames_file, 'w', encoding='utf-8') as f:
        for fname in clean_filenames:
            f.write(fname + '\n')
    print(f"💾 Saved: {word_filenames_file}")
    
    # ============================================
    # Build TFIDF-Lemm Matrix
    # ============================================
    
    lemm_matrix, lemm_features, lemm_vectorizer, lemm_stats = build_tfidf_matrix(
        documents=lemm_docs,
        filenames=lemm_filenames,
        matrix_name="TFIDF-Lemm",
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        use_bm25=True
    )
    
    # Save matrix
    lemm_matrix_file = Path(OUTPUT_FOLDER) / "tfidf_lemm_matrix.npz"
    save_npz(lemm_matrix_file, lemm_matrix)
    print(f"\n💾 Saved: {lemm_matrix_file}")
    
    # Save filenames mapping
    lemm_filenames_file = Path(OUTPUT_FOLDER) / "tfidf_lemm_filenames.txt"
    with open(lemm_filenames_file, 'w', encoding='utf-8') as f:
        for fname in lemm_filenames:
            f.write(fname + '\n')
    print(f"💾 Saved: {lemm_filenames_file}")
    
    # ============================================
    # Calculate Feature Importance
    # ============================================
    
    print("\n" + "="*70)
    print("📊 CALCULATING FEATURE IMPORTANCE")
    print("="*70)
    
    # For TFIDF-Word
    word_ig = calculate_information_gain(word_matrix, None, word_features)
    word_chi2 = calculate_chi_squared(word_matrix, None, word_features)
    
    # For TFIDF-Lemm
    lemm_ig = calculate_information_gain(lemm_matrix, None, lemm_features)
    lemm_chi2 = calculate_chi_squared(lemm_matrix, None, lemm_features)
    
    # ============================================
    # Export to Excel
    # ============================================
    
    excel_file = Path(OUTPUT_FOLDER) / "feature_importance.xlsx"
    export_to_excel(
        word_ig, word_chi2, lemm_ig, lemm_chi2,
        word_stats, lemm_stats, excel_file
    )
    
    # ============================================
    # Display Top Features
    # ============================================
    
    print("\n" + "="*70)
    print("🏆 TOP 20 FEATURES")
    print("="*70)
    
    print("\n📝 TFIDF-Word - Top 20 by Information Gain:")
    print(word_ig.head(20).to_string(index=False))
    
    print("\n📝 TFIDF-Word - Top 20 by Chi-squared:")
    print(word_chi2.head(20).to_string(index=False))
    
    print("\n📝 TFIDF-Lemm - Top 20 by Information Gain:")
    print(lemm_ig.head(20).to_string(index=False))
    
    print("\n📝 TFIDF-Lemm - Top 20 by Chi-squared:")
    print(lemm_chi2.head(20).to_string(index=False))
    
    # ============================================
    # Final Summary
    # ============================================
    
    print("\n" + "="*70)
    print("✅ COMPLETED!")
    print("="*70)
    print(f"\n📂 Output files:")
    print(f"   • {word_matrix_file}")
    print(f"   • {word_filenames_file}")
    print(f"   • {lemm_matrix_file}")
    print(f"   • {lemm_filenames_file}")
    print(f"   • {excel_file}")
    print(f"\n📊 Statistics:")
    print(f"   • TFIDF-Word: {word_stats['num_documents']} documents, {word_stats['num_features']} features")
    print(f"   • TFIDF-Lemm: {lemm_stats['num_documents']} documents, {lemm_stats['num_features']} features")
    print("\n🎉 TF-IDF matrices successfully created!")


if __name__ == "__main__":
    main()