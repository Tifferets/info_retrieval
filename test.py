import numpy as np
from scipy.sparse import load_npz

# For SPARSE matrices (TF-IDF)
matrix = load_npz('tf_idf_lemma_and_words/tfidf_word_matrix.npz')

print(f"Shape: {matrix.shape}")
print(f"Type: {type(matrix)}")
print(f"Sparsity: {(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100:.2f}%")

# View first document's vector
print(f"\nFirst document (first 10 features):")
print(matrix[1, :10].toarray())