"""
TF-IDF analysis module with GPU acceleration support.
Provides efficient text analysis for large-scale document collections.
"""

import pandas as pd
import numpy as np
from typing import Set, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class TfidfAnalyzer:
    """
    TF-IDF analyzer with optional GPU acceleration.
    """

    def __init__(
        self,
        min_df: int = 2,
        max_features: int = 2000,
        use_gpu: bool = True
    ):
        """
        Initialize TF-IDF analyzer.

        Args:
            min_df: Minimum document frequency for terms
            max_features: Maximum number of features to extract
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.min_df = min_df
        self.max_features = max_features
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.vectorizer = None
        self.feature_names = None

    def fit_transform(
        self,
        documents: pd.Series,
        filter_terms: Optional[Set[str]] = None,
        stopwords: Optional[Set[str]] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Fit vectorizer and transform documents to TF-IDF matrix.

        Args:
            documents: Series of text documents (space-separated tokens)
            filter_terms: Set of terms to exclude from features
            stopwords: Set of stopwords to exclude from vectorization
            verbose: Whether to print timing information

        Returns:
            DataFrame with TF-IDF scores (documents × terms)
        """
        start_time = time()

        # Vectorize documents
        # Note: We don't set token_pattern when using a custom tokenizer to avoid warnings
        vectorizer_params = {
            'tokenizer': lambda x: x.split(),
            'lowercase': False,
            'min_df': self.min_df,
            'max_features': self.max_features
        }

        if stopwords:
            vectorizer_params['stop_words'] = list(stopwords)

        self.vectorizer = TfidfVectorizer(**vectorizer_params)

        X = self.vectorizer.fit_transform(documents)
        feature_names = self.vectorizer.get_feature_names_out()

        # Filter unwanted terms
        if filter_terms:
            if self.use_gpu:
                X, feature_names = self._filter_gpu(X, feature_names, filter_terms)
            else:
                X, feature_names = self._filter_cpu(X, feature_names, filter_terms)

        self.feature_names = feature_names

        # Convert to DataFrame
        if self.use_gpu:
            X_array = cp.asnumpy(X.toarray()) if hasattr(X, 'toarray') else X.toarray()
        else:
            X_array = X.toarray()

        tfidf_df = pd.DataFrame(
            X_array,
            index=documents.index,
            columns=feature_names
        )

        if verbose:
            print(f"TF-IDF processing time: {time() - start_time:.2f} seconds")
            print(f"Shape: {tfidf_df.shape[0]} documents × {tfidf_df.shape[1]} terms")
            print(f"GPU acceleration: {'enabled' if self.use_gpu else 'disabled'}")

        return tfidf_df

    def _filter_gpu(self, X, feature_names, filter_terms):
        """Filter features using GPU acceleration."""
        X_gpu = cpsp.csr_matrix(X)
        valid_mask = cp.array([
            not any(fterm in term for fterm in filter_terms)
            for term in feature_names
        ])

        valid_indices = cp.where(valid_mask)[0]
        X_filtered = X_gpu[:, valid_indices]
        feature_names_filtered = feature_names[cp.asnumpy(valid_mask)]

        return X_filtered, feature_names_filtered

    def _filter_cpu(self, X, feature_names, filter_terms):
        """Filter features using CPU."""
        valid_mask = np.array([
            not any(fterm in term for fterm in filter_terms)
            for term in feature_names
        ])

        X_filtered = X[:, valid_mask]
        feature_names_filtered = feature_names[valid_mask]

        return X_filtered, feature_names_filtered

    def get_top_terms_per_document(
        self,
        tfidf_df: pd.DataFrame,
        top_n: int = 10,
        min_score: float = 0.0
    ) -> pd.DataFrame:
        """
        Extract top N terms for each document.

        Args:
            tfidf_df: TF-IDF DataFrame
            top_n: Number of top terms to extract per document
            min_score: Minimum TF-IDF score threshold

        Returns:
            DataFrame with columns: [document_id, token, tfidf]
        """
        top_terms = (
            tfidf_df.apply(lambda row: row.nlargest(top_n), axis=1)
            .stack()
            .reset_index()
            .rename(columns={
                'level_0': 'document_id',
                'level_1': 'token',
                0: 'tfidf'
            })
        )

        # Filter by minimum score
        if min_score > 0:
            top_terms = top_terms[top_terms['tfidf'] > min_score]

        return top_terms


def analyze_by_group(
    df: pd.DataFrame,
    group_col: str,
    text_col: str,
    analyzer: TfidfAnalyzer,
    top_n: int = 10,
    filter_terms: Optional[Set[str]] = None,
    stopwords: Optional[Set[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform TF-IDF analysis grouped by a column.

    Args:
        df: Input DataFrame with text data
        group_col: Column to group by (e.g., 'topic_title', 'author')
        text_col: Column containing space-separated tokens
        analyzer: TfidfAnalyzer instance
        top_n: Number of top terms per group
        filter_terms: Terms to exclude from analysis
        stopwords: Stopwords to exclude from vectorization

    Returns:
        Tuple of (tfidf_matrix, top_terms_df)
    """
    # Aggregate documents by group
    grouped_docs = df.groupby(group_col)[text_col].apply(lambda x: ' '.join(x))

    # Compute TF-IDF
    tfidf_df = analyzer.fit_transform(grouped_docs, filter_terms=filter_terms, stopwords=stopwords)

    # Extract top terms
    top_terms = analyzer.get_top_terms_per_document(tfidf_df, top_n=top_n)
    top_terms = top_terms.rename(columns={'document_id': group_col})

    return tfidf_df, top_terms
