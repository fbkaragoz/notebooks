import argparse
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from sqlite3 import Connection
from typing import cast

import numpy as np
import pandas as pd
from gensim.models import FastText


def preview_database(db_path: str | None, table_names: Iterable[str] | None = None) -> None:
    """
    Preview every table by showing the schema, five sample rows, and row counts.
    """
    conn: Connection = sqlite3.connect(db_path or ":memory:")
    cursor = conn.cursor()
    if table_names is None:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [row[0] for row in cursor.fetchall()]
    for table_name in table_names:
        print(f"\nTable: {table_name}")
        cursor.execute(f"PRAGMA table_info({table_name});")
        schema = cursor.fetchall()
        print("Schema:")
        for column in schema:
            print(f"  {column[1]} ({column[2]})")
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5;", conn)
        print("\nSample Rows:")
        print(df)
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        print(f"\nTotal Rows: {row_count}")
    conn.close()


def load_entries_with_topics(db_path: str, limit: int | None = None) -> pd.DataFrame:
    """
    Load the core entry fields and join topic metadata for downstream EDA/ML tasks.
    """
    query = """
        SELECT
            e.entry_id,
            e.topic_id,
            t.title AS topic_title,
            e.author_hash,
            e.favorites,
            e.created_at_ts,
            e.text_clean,
            e.crawl_ts
        FROM entries AS e
        LEFT JOIN topics AS t ON e.topic_id = t.topic_id
        WHERE e.text_clean IS NOT NULL AND TRIM(e.text_clean) <> ''
        ORDER BY e.created_at_ts
    """
    if limit:
        query += f"\n        LIMIT {int(limit)}"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    df["created_at_ts"] = pd.to_datetime(df["created_at_ts"], errors="coerce")
    df["crawl_ts"] = pd.to_datetime(df["crawl_ts"], errors="coerce")
    df["favorites"] = df["favorites"].fillna(0)
    df["entry_length_chars"] = df["text_clean"].str.len()
    df["entry_length_tokens"] = df["text_clean"].str.split().str.len()
    df["created_date"] = df["created_at_ts"].dt.date
    df["created_hour"] = df["created_at_ts"].dt.hour
    df["created_dayofweek"] = df["created_at_ts"].dt.day_name()
    return df


def report_basic_eda(df: pd.DataFrame) -> None:
    """
    Print headline metrics that help sanity-check the crawl before deeper analysis.
    """
    print("\n=== Basic Shape ===")
    print(f"Rows: {len(df):,}")
    print(f"Unique topics: {df['topic_id'].nunique():,}")
    print(f"Unique authors: {df['author_hash'].nunique():,}")

    print("\n=== Missing Values ===")
    missing = df[["topic_title", "author_hash", "favorites", "created_at_ts", "text_clean"]].isna().sum()
    print(missing.to_string())

    print("\n=== Entry Length (characters) ===")
    print(df["entry_length_chars"].describe(percentiles=[0.5, 0.75, 0.9, 0.99]).to_string())

    print("\n=== Favorites ===")
    print(df["favorites"].astype("int64").describe(percentiles=[0.5, 0.75, 0.9, 0.99]).to_string())

    print("\n=== Top Topics (by entry count) ===")
    print(df["topic_title"].value_counts().head(10).to_string())

    print("\n=== Top Authors (by entry count) ===")
    print(df["author_hash"].value_counts().head(10).to_string())

    duplicate_count = df["text_clean"].duplicated().sum()
    print(f"\n=== Duplicate Text Entries ===\nTotal duplicated texts (exact match): {duplicate_count:,}")


def train_baseline_favorites_classifier(df: pd.DataFrame, sample_size: int = 50_000) -> None:
    """
    Train a quick TF-IDF + Logistic Regression baseline to predict whether an entry is above-median favorites.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:  # pragma: no cover - depends on env setup
        print(f"Skipping baseline model (scikit-learn missing): {exc}")
        return

    working = df[["text_clean", "favorites"]].dropna()
    if working.empty:
        print("No rows with both text and favorites available for baseline training.")
        return

    threshold = working["favorites"].median()
    working = working.assign(target=(working["favorites"] > threshold).astype(int))

    unique_targets = working["target"].nunique()
    if unique_targets < 2:
        print(
            "Baseline skipped: favorites lack variance after thresholding. "
            "Consider a different target (e.g., favorites >= 1) or fix data collection."
        )
        return

    if sample_size and len(working) > sample_size:
        working = working.sample(sample_size, random_state=42)
        print(f"Sampled {len(working):,} rows for the baseline model (median favorites threshold: {threshold}).")
    else:
        print(f"Using all {len(working):,} rows for the baseline model (median favorites threshold: {threshold}).")

    X_train, X_test, y_train, y_test = train_test_split(
        working["text_clean"],
        working["target"],
        test_size=0.2,
        random_state=42,
        stratify=working["target"],
    )

    vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=5)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        solver="liblinear",
        max_iter=300,
        class_weight="balanced",
    )
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]

    print("\n=== Baseline Classification Report ===")
    print(classification_report(y_test, y_pred, digits=3))
    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC: {auc:.3f}")
    except ValueError:
        print("ROC AUC could not be calculated (only one class present in y_test).")


def prepare_topic_labels(
    df: pd.DataFrame,
    top_k: int = 20,
    other_label: str = "OTHER",
) -> pd.DataFrame:
    """
    Map infrequent topic titles into a shared OTHER bucket while preserving top-k topics.
    """
    topic_counts = df["topic_title"].value_counts()
    top_topics = topic_counts.nlargest(top_k).index
    df = df.copy()
    df["topic_bucket"] = np.where(df["topic_title"].isin(top_topics), df["topic_title"], other_label)
    return df


def train_topic_baseline(
    df: pd.DataFrame,
    top_k: int = 20,
    cutoff: str | None = None,
    min_df: int = 5,
    max_df: float = 0.6,
) -> None:
    """
    Train a LinearSVC on TF-IDF (word + char) features to predict coarse-grained topics.
    """
    try:
        from sklearn.base import TransformerMixin
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import classification_report, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        print(f"Skipping topic baseline (scikit-learn missing): {exc}")
        return

    dataset = df.dropna(subset=["topic_title", "text_clean", "created_at_ts"]).copy()
    if dataset.empty:
        print("Topic baseline skipped: no rows with topic_title, text_clean, and created_at_ts.")
        return

    dataset = prepare_topic_labels(dataset, top_k=top_k)

    if cutoff:
        cutoff_ts = pd.to_datetime(cutoff, errors="coerce")
        if pd.isna(cutoff_ts):
            print(f"Invalid cutoff '{cutoff}' supplied. Falling back to random split.")
            cutoff_ts = None
    else:
        cutoff_ts = None

    X = dataset["text_clean"]
    y = dataset["topic_bucket"]

    if cutoff_ts:
        train_mask = dataset["created_at_ts"] <= cutoff_ts
        test_mask = dataset["created_at_ts"] > cutoff_ts
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(
                "Cutoff produced an empty train/test split. Falling back to random 80/20 split."
            )
            cutoff_ts = None

    if cutoff_ts is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        cutoff_used = "Random 80/20 split"
    else:
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        cutoff_used = f"Time split @ {cutoff_ts.date()}"

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    word_vectorizer = cast(
        TransformerMixin,
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
        ),
    )
    char_vectorizer = cast(
        TransformerMixin,
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=5,
            max_df=0.8,
            sublinear_tf=True,
        ),
    )

    feature_union = FeatureUnion(
        transformer_list=[
            ("word", word_vectorizer),
            ("char", char_vectorizer),
        ],
        n_jobs=-1,
    )

    X_train_vec = feature_union.fit_transform(X_train)
    X_test_vec = feature_union.transform(X_test)

    model = LinearSVC()
    model.fit(X_train_vec, y_train_enc)

    y_pred_enc = model.predict(X_test_vec)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    micro_f1 = f1_score(y_test_enc, y_pred_enc, average="micro")
    macro_f1 = f1_score(y_test_enc, y_pred_enc, average="macro")

    print("\n=== Topic Classification Baseline ===")
    print(f"Split: {cutoff_used}")
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    print("Label distribution (train):")
    print(y_train.value_counts().sort_values(ascending=False).to_string())
    print("Label distribution (test):")
    print(y_test.value_counts().sort_values(ascending=False).to_string())
    print(f"\nMicro F1: {micro_f1:.3f} | Macro F1: {macro_f1:.3f}")
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, digits=3))


def run_topic_clustering(
    df: pd.DataFrame,
    sample_size: int = 20_000,
    min_cluster_size: int = 50,
    min_samples: int = 10,
) -> None:
    """
    Run UMAP + HDBSCAN clustering on a TF-IDF representation and print cluster samples.
    """
    try:
        from cuml.cluster.hdbscan import HDBSCAN as CuHDBSCAN
        from cuml.feature_extraction.text import TfidfVectorizer as CuTfidfVectorizer
        from cuml.manifold import UMAP as CuUMAP
        import cupy as cp

        use_gpu = True
    except ImportError:
        use_gpu = False

    if not use_gpu:
        try:
            import hdbscan
            import umap
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            print(f"Skipping clustering (missing dependency): {exc}")
            print("Install umap-learn and hdbscan to enable clustering.")
            return

    corpus = df.dropna(subset=["text_clean"]).copy()
    if corpus.empty:
        print("Clustering skipped: no entries with text_clean.")
        return

    if sample_size and len(corpus) > sample_size:
        corpus = corpus.sample(sample_size, random_state=42)
        print(f"Sampling {len(corpus):,} rows for clustering.")

    if use_gpu:
        print("Running clustering with GPU acceleration (cuML).")
        vectorizer = CuTfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=5,
            max_features=40_000,
            sublinear_tf=True,
        )
        tfidf = vectorizer.fit_transform(corpus["text_clean"])

        reducer = CuUMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=10,
            metric="cosine",
            random_state=42,
        )
        embeddings_gpu = reducer.fit_transform(tfidf)
        embeddings = cp.asnumpy(embeddings_gpu)

        clusterer = CuHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels_gpu = clusterer.fit_predict(embeddings_gpu)
        labels = cp.asnumpy(labels_gpu)
    else:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=5,
            max_features=40_000,
            sublinear_tf=True,
        )
        tfidf = vectorizer.fit_transform(corpus["text_clean"])

        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.0,
            n_components=10,
            metric="cosine",
            random_state=42,
        )
        embeddings = reducer.fit_transform(tfidf)
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]
        embeddings = np.asarray(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings)

    corpus = corpus.assign(cluster=labels)
    clustered = corpus[corpus["cluster"] >= 0]
    noise = corpus[corpus["cluster"] < 0]

    if clustered.empty:
        print("HDBSCAN returned only noise. Consider lowering min_cluster_size or using a different embedding.")
        return

    print("\n=== HDBSCAN Clusters ===")
    cluster_counts = clustered["cluster"].value_counts().sort_values(ascending=False)
    print("Cluster sizes:")
    print(cluster_counts.to_string())
    print(f"Noise points: {len(noise):,}")

    print("\nSample texts per cluster (first 3 entries):")
    for cluster_id in cluster_counts.index[:10]:
        sample_texts = clustered[clustered["cluster"] == cluster_id]["text_clean"].head(3).tolist()
        print(f"\nCluster {cluster_id} (size={cluster_counts[cluster_id]}):")
        for idx, text in enumerate(sample_texts, 1):
            snippet = text[:280].replace("\n", " ")
            print(f"  {idx}. {snippet}{'...' if len(text) > 280 else ''}")


def analyze_topic_trends(
    df: pd.DataFrame,
    top_k: int = 20,
    freq: str = "W",
    zscore_threshold: float = 3.0,
) -> pd.DataFrame | None:
    """
    Aggregate topic counts over time and flag sharp spikes using z-scores.
    """
    if df["created_at_ts"].isna().all():
        print("Trend analysis skipped: no created_at_ts values available.")
        return None

    dataset = df.dropna(subset=["topic_title", "created_at_ts"]).copy()
    dataset = prepare_topic_labels(dataset, top_k=top_k)
    dataset["period"] = dataset["created_at_ts"].dt.to_period(freq).dt.to_timestamp()

    counts_raw = (
        dataset.groupby(["period", "topic_bucket"])
        .size()
        .rename("entry_count")
        .reset_index()
    )
    counts_by_topic = counts_raw.sort_values(["topic_bucket", "period"]).reset_index(drop=True)

    print("\n=== Topic Timeline (oldest → newest snapshot) ===")
    oldest_snapshot = counts_by_topic.groupby("topic_bucket").head(5)
    print(oldest_snapshot.sort_values(["topic_bucket", "period"]).to_string(index=False))

    print("\n=== Topic Timeline (most recent periods) ===")
    recent_snapshot = counts_by_topic.groupby("topic_bucket").tail(5)
    print(recent_snapshot.sort_values(["topic_bucket", "period"]).to_string(index=False))

    counts_stats = counts_by_topic.copy()
    counts_stats["zscore"] = counts_stats.groupby("topic_bucket")["entry_count"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else 0
    )
    counts_stats["abs_zscore"] = counts_stats["zscore"].abs()
    counts_stats["is_anomaly"] = counts_stats["abs_zscore"] >= zscore_threshold
    anomalies = counts_stats[counts_stats["is_anomaly"]]
    if anomalies.empty:
        print("No anomalies detected at the current z-score threshold.")
    else:
        print("\n=== Detected Topic Spikes ===")
        print(
            anomalies.sort_values(
                ["abs_zscore", "period"], ascending=[False, True]
            ).to_string(index=False)
        )

    trend_counts = counts_stats.sort_values(["period", "topic_bucket"]).reset_index(drop=True)
    return trend_counts

# === From Word to Classfier === #

"""The overall workflow will be as follows:
1. Convert the sentences into tokens
2. Train a FastText model on the tokens
3. Convert the vectors into a format suitable for classification
4. Train a classifier on the vectors"""

def prepare_sentences(entries_df: pd.DataFrame) -> list[list[str]]:
    """
    Convert the text_clean column into a list of tokenized sentences for training.

    Args:
        entries_df: DataFrame containing the text_clean column

    Returns:
        List of tokenized sentences (list of token lists)
    """
    try:
        from .text_utils import tokenize
    except ImportError:
        # Fallback for direct script execution
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        from text_utils import tokenize

    # Tokenize each entry
    tokenized_sentences = [tokenize(text) for text in entries_df['text_clean'].tolist()]
    # Filter out empty token lists
    tokenized_sentences = [sent for sent in tokenized_sentences if sent]
    return tokenized_sentences


def train_fasttext_model(
    sentences: list[list[str]],
    vector_size: int = 150,
    window: int = 5,
    min_count: int = 5,
    workers: int = 4,
    sg: int = 1,
    epochs: int = 10,
) -> FastText:
    """
    Train a FastText model on tokenized sentences using Gensim.

    Args:
        sentences: List of tokenized sentences (each sentence is a list of tokens)
        vector_size: Dimensionality of word vectors
        window: Maximum distance between current and predicted word
        min_count: Minimum word frequency threshold
        workers: Number of worker threads
        sg: Training algorithm (1 for skip-gram, 0 for CBOW)
        epochs: Number of training epochs

    Returns:
        Trained FastText model
    """
    try:
        from gensim.models import FastText
    except ImportError as exc:
        raise ImportError("gensim is required for FastText training. Install it with: pip install gensim") from exc

    print(f"Training FastText model on {len(sentences):,} sentences...")
    print(f"Parameters: vector_size={vector_size}, window={window}, min_count={min_count}, epochs={epochs}")

    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )

    print(f"Model trained. Vocabulary size: {len(model.wv):,}")
    return model


def create_document_vectors(
    entries_df: pd.DataFrame,
    fasttext_model: FastText,
    tokenized_sentences: list[list[str]] | None = None,
) -> np.ndarray:
    """
    Create document vectors by averaging word vectors from the FastText model.

    Args:
        entries_df: DataFrame containing the text_clean column
        fasttext_model: Trained FastText model
        tokenized_sentences: Pre-tokenized sentences (optional, will tokenize if not provided)

    Returns:
        Numpy array of document vectors (n_documents, vector_size)
    """
    try:
        from .text_utils import tokenize
    except ImportError:
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        from text_utils import tokenize

    if tokenized_sentences is None:
        tokenized_sentences = [tokenize(text) for text in entries_df['text_clean'].tolist()]

    print(f"Creating document vectors for {len(tokenized_sentences):,} documents...")

    document_vectors = []
    for tokens in tokenized_sentences:
        if not tokens:
            # If no tokens, create zero vector
            document_vectors.append(np.zeros(fasttext_model.vector_size))
        else:
            # Average word vectors for all tokens in the document
            word_vectors = [fasttext_model.wv[token] for token in tokens if token in fasttext_model.wv]
            if word_vectors:
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                # If no tokens are in vocabulary, create zero vector
                doc_vector = np.zeros(fasttext_model.vector_size)
            document_vectors.append(doc_vector)

    document_vectors_array = np.array(document_vectors)
    print(f"Document vectors created. Shape: {document_vectors_array.shape}")
    return document_vectors_array


def train_fasttext_classifier(
    df: pd.DataFrame,
    top_k: int = 20,
    vector_size: int = 150,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Train a classifier on FastText document vectors to predict topic categories.

    This function:
    1. Prepares and tokenizes sentences
    2. Trains a FastText word embedding model
    3. Creates document vectors by averaging word vectors
    4. Trains a Logistic Regression classifier on the document vectors

    Args:
        df: DataFrame containing text_clean and topic_title columns
        top_k: Number of top topics to classify (rest become "OTHER")
        vector_size: Dimensionality of FastText word vectors
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        print(f"Skipping FastText classifier (scikit-learn missing): {exc}")
        return

    # Step 1: Prepare data
    dataset = df.dropna(subset=["topic_title", "text_clean"]).copy()
    if dataset.empty:
        print("FastText classifier skipped: no rows with topic_title and text_clean.")
        return

    dataset = prepare_topic_labels(dataset, top_k=top_k)
    # Reset index to ensure we can track which rows have valid tokens
    dataset = dataset.reset_index(drop=True)

    print(f"\n=== FastText Topic Classification ===")
    print(f"Dataset size: {len(dataset):,} entries")
    print(f"Number of classes: {dataset['topic_bucket'].nunique()}")

    # Step 2: Tokenize sentences and track valid indices
    try:
        from .text_utils import tokenize
    except ImportError:
        import sys
        from pathlib import Path
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        from text_utils import tokenize

    # Tokenize and keep track of which entries have valid tokens
    tokenized_data = []
    valid_indices = []
    for idx, text in enumerate(dataset['text_clean'].tolist()):
        tokens = tokenize(text)
        if tokens:  # Only keep entries with non-empty tokens
            tokenized_data.append(tokens)
            valid_indices.append(idx)

    # Filter dataset to only include entries with valid tokens
    dataset = dataset.iloc[valid_indices].reset_index(drop=True)
    tokenized_sentences = tokenized_data

    print(f"Tokenized {len(tokenized_sentences):,} sentences (filtered {len(df) - len(tokenized_sentences):,} empty entries)")

    # Step 3: Train FastText model
    fasttext_model = train_fasttext_model(
        sentences=tokenized_sentences,
        vector_size=vector_size,
    )

    # Step 4: Create document vectors (now dataset and tokenized_sentences are aligned)
    print(f"Creating document vectors for {len(tokenized_sentences):,} documents...")
    document_vectors = []
    for tokens in tokenized_sentences:
        # Average word vectors for all tokens in the document
        word_vectors = [fasttext_model.wv[token] for token in tokens if token in fasttext_model.wv]
        if word_vectors:
            doc_vector = np.mean(word_vectors, axis=0)
        else:
            # If no tokens are in vocabulary, create zero vector
            doc_vector = np.zeros(fasttext_model.vector_size)
        document_vectors.append(doc_vector)

    document_vectors = np.array(document_vectors)
    print(f"Document vectors created. Shape: {document_vectors.shape}")

    # Step 5: Prepare labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataset['topic_bucket'])

    # Step 6: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        document_vectors,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"\nTrain size: {len(X_train):,}, Test size: {len(X_test):,}")

    # Step 7: Train classifier
    print("Training Logistic Regression classifier...")
    classifier = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )
    classifier.fit(X_train, y_train)

    # Step 8: Evaluate
    y_pred = classifier.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)

    micro_f1 = f1_score(y_test, y_pred, average="micro")
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nMicro F1: {micro_f1:.3f} | Macro F1: {macro_f1:.3f}")
    print("\nDetailed classification report:")
    print(classification_report(y_test_labels, y_pred_labels, digits=3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA and baseline modeling utilities for eksisozluk crawl.")
    parser.add_argument("--db-path", default="../datasets/eksi.db", help="Path to the SQLite database file.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional LIMIT when reading from SQLite.")
    parser.add_argument(
        "--baseline-sample",
        type=int,
        default=50_000,
        help="Number of rows to sample for the baseline classifier (set 0 to use all).",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Train a quick TF-IDF + Logistic Regression model to predict above-median favorites.",
    )
    parser.add_argument(
        "--run-topic-baseline",
        action="store_true",
        help="Train a TF-IDF + LinearSVC model to classify entries into top-k topics.",
    )
    parser.add_argument("--topic-top-k", type=int, default=20, help="Number of dominant topics to keep before bucketing.")
    parser.add_argument(
        "--time-cutoff",
        type=str,
        default=None,
        help="ISO date (YYYY-MM-DD) to use as train/test cutoff for the topic classifier.",
    )
    parser.add_argument(
        "--run-clustering",
        action="store_true",
        help="Run UMAP + HDBSCAN clustering on sample entries to surface latent topics.",
    )
    parser.add_argument(
        "--run-trend-analysis",
        action="store_true",
        help="Aggregate topic counts over time and flag spikes.",
    )
    parser.add_argument(
        "--run-fasttext-classifier",
        action="store_true",
        help="Train a FastText + Logistic Regression model to classify entries into top-k topics.",
    )
    parser.add_argument(
        "--fasttext-vector-size",
        type=int,
        default=150,
        help="Dimensionality of FastText word vectors.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Show table schemas/sample rows instead of loading the joined entries dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview_only:
        preview_database(args.db_path)
        return

    df = load_entries_with_topics(args.db_path, limit=args.sample_limit)
    report_basic_eda(df)

    if args.run_baseline:
        train_baseline_favorites_classifier(df, sample_size=args.baseline_sample)

    if args.run_topic_baseline:
        train_topic_baseline(
            df,
            top_k=args.topic_top_k,
            cutoff=args.time_cutoff,
        )

    if args.run_clustering:
        run_topic_clustering(df)

    if args.run_trend_analysis:
        analyze_topic_trends(df, top_k=args.topic_top_k)

    if args.run_fasttext_classifier:
        train_fasttext_classifier(
            df,
            top_k=args.topic_top_k,
            vector_size=args.fasttext_vector_size,
        )


if __name__ == "__main__":
    main()
    
