"""
Utilities for exploring Ekşi Sözlük reference graphs.

Example:
    from pyscripts.reference_analysis import (
        load_references_map,
        reference_type_counts,
        build_topic_reference_edges,
    )

These helpers make it easy to:
* Load the `references_map` table into pandas.
* Summarise reference usage by type and per-entry.
* Build topic-to-topic edge tables (suitable for NetworkX or further EDA).
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import pandas as pd

if TYPE_CHECKING:
    import networkx as nx

__all__ = [
    "load_references_map",
    "reference_type_counts",
    "entry_reference_features",
    "build_topic_reference_edges",
    "build_topic_reference_graph",
]


def load_references_map(db_path: str, limit: int | None = None) -> pd.DataFrame:
    """
    Read the references_map table and return a DataFrame with friendly dtypes.
    """
    query = """
        SELECT
            entry_id,
            ref_type,
            target_kind,
            target_id,
            target_slug,
            target_url,
            anchor_text,
            char_start,
            char_end,
            created_at_ts
        FROM references_map
        ORDER BY entry_id
    """
    if limit:
        query += f"\n        LIMIT {int(limit)}"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    df["created_at_ts"] = pd.to_datetime(df["created_at_ts"], errors="coerce")
    for col in ("char_start", "char_end"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["target_slug"] = df["target_slug"].fillna("").str.strip()
    df["target_slug_normalized"] = df["target_slug"].str.lower().replace({"": pd.NA})

    def _extract_domain(url: str | None) -> str | None:
        if not url:
            return None
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        return host or None

    df["target_domain"] = df["target_url"].apply(_extract_domain)
    return df


def reference_type_counts(ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count references by ref_type/target_kind to see the mix of internal/external links.
    """
    if ref_df.empty:
        return pd.DataFrame(columns=["ref_type", "target_kind", "reference_count"])

    counts = (
        ref_df.groupby(["ref_type", "target_kind"])
        .size()
        .reset_index(name="reference_count")
        .sort_values("reference_count", ascending=False)
        .reset_index(drop=True)
    )
    return counts


def entry_reference_features(ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-entry reference counts and diversity metrics.

    Returns a DataFrame keyed by entry_id.
    """
    if ref_df.empty:
        return pd.DataFrame(
            columns=[
                "entry_id",
                "total_references",
                "internal_ref_count",
                "external_ref_count",
                "unique_target_topics",
                "unique_external_domains",
            ]
        )

    counts = (
        ref_df.groupby(["entry_id", "ref_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["internal", "external"], fill_value=0)
        .rename(columns={"internal": "internal_ref_count", "external": "external_ref_count"})
        .reset_index()
    )
    counts["total_references"] = counts["internal_ref_count"] + counts["external_ref_count"]

    topic_diversity = (
        ref_df[ref_df["target_kind"] == "topic"]
        .groupby("entry_id")["target_slug_normalized"]
        .nunique()
        .rename("unique_target_topics")
    )

    domain_diversity = (
        ref_df[ref_df["ref_type"] == "external"]
        .groupby("entry_id")["target_domain"]
        .nunique()
        .rename("unique_external_domains")
    )

    features = (
        counts.merge(topic_diversity, on="entry_id", how="left")
        .merge(domain_diversity, on="entry_id", how="left")
        .fillna({"unique_target_topics": 0, "unique_external_domains": 0})
    )
    features["unique_target_topics"] = features["unique_target_topics"].astype(int)
    features["unique_external_domains"] = features["unique_external_domains"].astype(int)
    return features


def build_topic_reference_edges(
    entries_df: pd.DataFrame,
    references_df: pd.DataFrame,
    topics_df: pd.DataFrame | None = None,
    include_unresolved: bool = False,
    min_weight: int = 1,
) -> pd.DataFrame:
    """
    Aggregate topic-to-topic reference edges (source topic -> referenced topic/slug).
    """
    if entries_df.empty or references_df.empty:
        return pd.DataFrame(
            columns=[
                "source_topic_id",
                "source_topic_title",
                "target_kind",
                "target_slug",
                "target_topic_id",
                "target_topic_title",
                "edge_weight",
            ]
        )

    refs = references_df[references_df["ref_type"] == "internal"].copy()
    if not include_unresolved:
        refs = refs[refs["target_kind"] == "topic"]
    else:
        refs = refs[refs["target_kind"].isin(["topic", "unresolved"])]

    refs = refs.dropna(subset=["target_slug_normalized"])
    if refs.empty:
        return pd.DataFrame(
            columns=[
                "source_topic_id",
                "source_topic_title",
                "target_kind",
                "target_slug",
                "target_topic_id",
                "target_topic_title",
                "edge_weight",
            ]
        )

    entry_topics = entries_df[["entry_id", "topic_id", "topic_title"]].dropna(subset=["entry_id", "topic_id"])
    entry_topics = entry_topics.rename(
        columns={"topic_id": "source_topic_id", "topic_title": "source_topic_title"}
    )

    joined = refs.merge(entry_topics, on="entry_id", how="inner")
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "source_topic_id",
                "source_topic_title",
                "target_kind",
                "target_slug",
                "target_topic_id",
                "target_topic_title",
                "edge_weight",
            ]
        )

    edges = joined.copy()
    edges["target_topic_id"] = pd.NA
    edges["target_topic_title"] = pd.NA

    if topics_df is not None and not topics_df.empty and "slug" in topics_df.columns:
        topics_norm = topics_df.copy()
        topics_norm["slug_normalized"] = topics_norm["slug"].fillna("").str.strip().str.lower()
        topics_norm = topics_norm[["topic_id", "title", "slug_normalized"]]
        mapped = edges.merge(
            topics_norm,
            left_on="target_slug_normalized",
            right_on="slug_normalized",
            how="left",
        )
        edges["target_topic_id"] = mapped["topic_id"]
        edges["target_topic_title"] = mapped["title"]

    group_cols: list[str] = [
        "source_topic_id",
        "source_topic_title",
        "target_kind",
        "target_slug",
        "target_topic_id",
        "target_topic_title",
    ]

    aggregated = (
        edges.groupby(group_cols)
        .size()
        .reset_index(name="edge_weight")
        .sort_values("edge_weight", ascending=False)
        .reset_index(drop=True)
    )

    if min_weight > 1:
        aggregated = aggregated[aggregated["edge_weight"] >= min_weight].reset_index(drop=True)
    return aggregated


def build_topic_reference_graph(
    edges_df: pd.DataFrame,
    weight_col: str = "edge_weight",
) -> "nx.DiGraph":
    """
    Turn an edge table into a NetworkX DiGraph for downstream network analysis.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install networkx to construct the reference graph.") from exc

    graph = nx.DiGraph()
    if edges_df.empty:
        return graph

    for _, row in edges_df.iterrows():
        src = row.get("source_topic_id")
        dst = row.get("target_topic_id") or row.get("target_slug")
        if not src or not dst:
            continue

        graph.add_node(
            src,
            title=row.get("source_topic_title"),
        )
        graph.add_node(
            dst,
            title=row.get("target_topic_title"),
            slug=row.get("target_slug"),
            target_kind=row.get("target_kind"),
        )

        weight = row.get(weight_col, 1)
        graph.add_edge(src, dst, weight=weight)
    return graph
