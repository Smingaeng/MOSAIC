"""Build optional community conformity embeddings for CoMoE.

Default mode aggregates existing user/item conformity LLM embeddings and appends
community-level scalar conformity statistics from scalar_popularity_feature.pkl.
It also writes community_conformity_summary.jsonl.  If --openai-model is supplied,
summary texts are embedded with the OpenAI embedding API instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from community_utils import data_dir_for, detect_communities, load_pickle, load_train_matrix, save_pickle  # noqa: E402


def load_embedding(path: Path, expected_rows: int, name: str) -> np.ndarray:
    arr = np.asarray(load_pickle(path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D; got shape={arr.shape}")
    if arr.shape[0] != expected_rows:
        raise ValueError(f"{name} row mismatch: expected {expected_rows}, got {arr.shape[0]}")
    return arr.astype(np.float32, copy=False)


def rank_popularity_high_is_popular(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts).reshape(-1)
    out = np.zeros_like(counts, dtype=np.float32)
    active = np.where(counts > 0)[0]
    if active.size == 0:
        return out
    active_counts = counts[active]
    unique_counts = np.unique(active_counts)[::-1]
    if unique_counts.size == 1:
        out[active] = 1.0
        return out
    for rank, cnt in enumerate(unique_counts):
        out[active[active_counts == cnt]] = 1.0 - float(rank) / float(unique_counts.size - 1)
    return out.astype(np.float32)


def load_or_detect_community(args, mat: sp.csr_matrix) -> Dict:
    root = data_dir_for(args.dataset, args.data_dir)
    if args.community_path:
        path = Path(args.community_path)
        if path.exists() and not args.rebuild_community:
            return load_pickle(path)
    else:
        scalar_path = root / "scalar_popularity_feature.pkl"
        if scalar_path.exists() and not args.rebuild_community:
            scalar = load_pickle(scalar_path)
            if "user_community" in scalar:
                return {
                    "dataset": args.dataset,
                    "user_community": np.asarray(scalar["user_community"], dtype=np.int64),
                    "item_community": np.asarray(scalar.get("item_community", []), dtype=np.int64),
                    "num_communities": int(scalar.get("num_communities", np.asarray(scalar["user_community"]).max() + 1)),
                    "metadata": {"source": str(scalar_path)},
                }
        path = root / "community_assignments.pkl"
        if path.exists() and not args.rebuild_community:
            return load_pickle(path)

    community = detect_communities(
        dataset=args.dataset,
        data_dir=args.data_dir,
        method=args.community_method,
        seed=args.seed,
        barber_clusters=args.barber_clusters,
        barber_max_iter=args.barber_max_iter,
    )
    save_pickle(community, path)
    print(f"[DONE] Saved community assignments: {path}")
    return community


def compute_scalar_stats(mat: sp.csr_matrix, user_comm: np.ndarray) -> Dict[str, np.ndarray]:
    mat = (mat != 0).astype(np.float32).tocsr()
    num_c = int(user_comm.max()) + 1 if user_comm.size else 0
    item_num = mat.shape[1]
    global_item_counts = np.asarray(mat.sum(axis=0)).reshape(-1).astype(np.float32)
    global_item_rank = rank_popularity_high_is_popular(global_item_counts)
    local_counts = np.zeros((num_c, item_num), dtype=np.float32)
    coo = mat.tocoo()
    np.add.at(local_counts, (user_comm[coo.row], coo.col), 1.0)
    local_rank = np.zeros_like(local_counts, dtype=np.float32)
    for cid in range(num_c):
        local_rank[cid] = rank_popularity_high_is_popular(local_counts[cid])

    local_user_conf = np.zeros(mat.shape[0], dtype=np.float32)
    global_user_conf = np.zeros(mat.shape[0], dtype=np.float32)
    for u in range(mat.shape[0]):
        start, end = mat.indptr[u], mat.indptr[u + 1]
        if end <= start:
            continue
        items = mat.indices[start:end]
        cid = int(user_comm[u])
        global_user_conf[u] = float(global_item_rank[items].mean())
        local_user_conf[u] = float(local_rank[cid, items].mean())

    stats = np.zeros((num_c, 10), dtype=np.float32)
    for cid in range(num_c):
        users = np.where(user_comm == cid)[0]
        active = local_counts[cid] > 0
        lvals = local_rank[cid, active]
        gvals = global_item_rank[active]
        uvals = local_user_conf[users] if users.size else np.array([], dtype=np.float32)
        stats[cid, 0] = float(users.size)
        stats[cid, 1] = float(active.sum())
        if lvals.size:
            stats[cid, 2] = float(lvals.mean())
            stats[cid, 3] = float(lvals.std())
            stats[cid, 4] = float((lvals >= 0.99).mean())
            stats[cid, 5] = float((lvals >= 0.95).mean())
            stats[cid, 6] = float((lvals <= 0.20).mean())
            # Positive means local popularity is stronger than global popularity.
            stats[cid, 7] = float((lvals - gvals).mean())
        if uvals.size:
            stats[cid, 8] = float(uvals.mean())
            stats[cid, 9] = float(uvals.std())
    return {
        "global_item_rankpop": global_item_rank,
        "local_item_rankpop": local_rank,
        "local_user_conformity": local_user_conf,
        "global_user_conformity": global_user_conf,
        "community_conformity_stats": stats,
        "local_counts": local_counts,
    }


def load_or_compute_scalar(args, mat: sp.csr_matrix, user_comm: np.ndarray) -> Dict[str, np.ndarray]:
    root = data_dir_for(args.dataset, args.data_dir)
    path = Path(args.scalar_path) if args.scalar_path else root / "scalar_popularity_feature.pkl"
    candidates = [path]
    if args.scalar_path is None:
        candidates.append(root / "scalar_popularity_features.pkl")
    for candidate in candidates:
        if candidate.exists():
            obj = load_pickle(candidate)
            if isinstance(obj, dict) and "community_conformity_stats" in obj:
                return obj
    return compute_scalar_stats(mat, user_comm)


def build_default_embeddings(
    mat: sp.csr_matrix,
    user_comm: np.ndarray,
    user_conf: np.ndarray,
    item_conf: np.ndarray,
    scalar: Dict[str, np.ndarray],
    user_weight: float,
    item_weight: float,
    append_stats: bool,
) -> np.ndarray:
    if user_conf.shape[1] != item_conf.shape[1]:
        raise ValueError(
            "Default centroid mode requires user and item conformity embeddings to have the same dimension. "
            "Use --openai-model to embed text summaries if their dimensions differ."
        )
    num_c = int(user_comm.max()) + 1 if user_comm.size else 0
    stats = np.asarray(scalar.get("community_conformity_stats", np.zeros((num_c, 10))), dtype=np.float32)
    if stats.shape[0] != num_c:
        stats = np.zeros((num_c, 10), dtype=np.float32)
    base_dim = user_conf.shape[1]
    out_dim = base_dim + (stats.shape[1] if append_stats else 0)
    out = np.zeros((num_c, out_dim), dtype=np.float32)
    mat = mat.tocsr()
    for cid in range(num_c):
        users = np.where(user_comm == cid)[0]
        if users.size:
            u_cent = user_conf[users].mean(axis=0)
        else:
            u_cent = np.zeros(base_dim, dtype=np.float32)
        counts = np.asarray(mat[users].sum(axis=0)).reshape(-1) if users.size else np.zeros(mat.shape[1])
        active = np.where(counts > 0)[0]
        if active.size:
            i_cent = np.average(item_conf[active], axis=0, weights=counts[active].astype(np.float32))
        else:
            i_cent = np.zeros(base_dim, dtype=np.float32)
        denom = max(1e-8, user_weight + item_weight)
        out[cid, :base_dim] = (user_weight * u_cent + item_weight * i_cent) / denom
        if append_stats:
            out[cid, base_dim:] = stats[cid]
    return out.astype(np.float32)


def top_items_by_metric(values: np.ndarray, counts: np.ndarray, top_n: int, descending: bool = True) -> List[Dict]:
    active = np.where(counts > 0)[0]
    if active.size == 0:
        return []
    order = active[np.argsort(-values[active] if descending else values[active])]
    order = order[:top_n]
    return [
        {"iid": int(iid), "rankpop": float(values[iid]), "train_interactions_in_community": int(counts[iid])}
        for iid in order
    ]


def write_summaries(output_path: Path, mat: sp.csr_matrix, user_comm: np.ndarray, scalar: Dict[str, np.ndarray], top_n: int) -> List[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_rank = np.asarray(scalar["local_item_rankpop"], dtype=np.float32)
    local_counts = np.asarray(scalar.get("local_counts"), dtype=np.float32) if "local_counts" in scalar else None
    if local_counts is None or local_counts.shape != local_rank.shape:
        local_counts = np.zeros_like(local_rank, dtype=np.float32)
        coo = mat.tocoo()
        np.add.at(local_counts, (user_comm[coo.row], coo.col), 1.0)
    local_user_conf = np.asarray(scalar.get("local_user_conformity", np.zeros(mat.shape[0])), dtype=np.float32)
    stats = np.asarray(scalar.get("community_conformity_stats", np.zeros((local_rank.shape[0], 10))), dtype=np.float32)
    texts = []
    with output_path.open("w", encoding="utf-8") as f:
        for cid in range(local_rank.shape[0]):
            users = np.where(user_comm == cid)[0]
            popular = top_items_by_metric(local_rank[cid], local_counts[cid], top_n, descending=True)
            niche = top_items_by_metric(local_rank[cid], local_counts[cid], top_n, descending=False)
            user_mean = float(local_user_conf[users].mean()) if users.size else 0.0
            user_std = float(local_user_conf[users].std()) if users.size else 0.0
            stat_list = stats[cid].astype(float).tolist() if cid < stats.shape[0] else []
            text = (
                f"Community {cid} conformity profile. "
                f"Number of users: {users.size}. "
                f"Mean local user conformity: {user_mean:.4f}; std: {user_std:.4f}. "
                f"Scalar stats: {stat_list}. "
                f"Most locally popular items: {json.dumps(popular, ensure_ascii=False)}. "
                f"Most niche active items: {json.dumps(niche, ensure_ascii=False)}. "
                "Summarize whether this community follows global popularity, local trends, mainstream items, or niche discoveries."
            )
            record = {
                "cid": cid,
                "num_users": int(users.size),
                "mean_local_user_conformity": user_mean,
                "std_local_user_conformity": user_std,
                "community_conformity_stats": stat_list,
                "top_local_popular_items": popular,
                "top_local_niche_items": niche,
                "summary_text_for_embedding": text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            texts.append(text)
    print(f"[DONE] Saved community conformity summaries: {output_path}")
    return texts


def batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embed_with_openai(texts: List[str], model: str, batch_size: int, max_retries: int) -> np.ndarray:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before using --openai-model.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("The openai package is required for --openai-model.") from exc
    client = OpenAI()
    chunks = []
    done = 0
    for batch in batched(texts, batch_size):
        retry = 0
        while True:
            try:
                response = client.embeddings.create(model=model, input=list(batch))
                data = sorted(response.data, key=lambda x: x.index)
                chunks.append(np.asarray([x.embedding for x in data], dtype=np.float32))
                done += len(batch)
                print(f"[INFO] OpenAI embedded {done:,}/{len(texts):,} community conformity summaries")
                break
            except Exception:
                retry += 1
                if retry > max_retries:
                    raise
                time.sleep(min(60, 2 ** retry))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CoMoE community conformity embeddings.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--community-path", default=None)
    parser.add_argument("--community-method", default="auto", choices=["auto", "louvain", "barber"])
    parser.add_argument("--rebuild-community", action="store_true")
    parser.add_argument("--barber-clusters", type=int, default=None)
    parser.add_argument("--barber-max-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--user-conformity", default=None, help="Default: data/<dataset>/user_conf_emb.pkl")
    parser.add_argument("--item-conformity", default=None, help="Default: data/<dataset>/item_conf_emb.pkl")
    parser.add_argument("--scalar-path", default=None, help="Default: data/<dataset>/scalar_popularity_feature.pkl")
    parser.add_argument("--output", default=None, help="Default: data/<dataset>/community_conf_emb.pkl")
    parser.add_argument("--summary-output", default=None, help="Default: data/<dataset>/community_conformity_summary.jsonl")
    parser.add_argument("--top-items", type=int, default=30)
    parser.add_argument("--user-weight", type=float, default=0.5)
    parser.add_argument("--item-weight", type=float, default=0.5)
    parser.add_argument("--no-append-stats", action="store_true", help="Do not append scalar conformity stats to centroid vectors.")
    parser.add_argument("--openai-model", default=None, help="Optional OpenAI embedding model for summary text.")
    parser.add_argument("--openai-batch-size", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = data_dir_for(args.dataset, args.data_dir)
    output_path = Path(args.output) if args.output else root / "community_conf_emb.pkl"
    summary_path = Path(args.summary_output) if args.summary_output else root / "community_conformity_summary.jsonl"
    if output_path.exists() and not args.overwrite:
        print(f"[SKIP] {output_path} exists. Use --overwrite to rebuild.")
        return

    mat = load_train_matrix(args.dataset, args.data_dir)
    community = load_or_detect_community(args, mat)
    user_comm = np.asarray(community["user_community"], dtype=np.int64)
    user_path = Path(args.user_conformity) if args.user_conformity else root / "user_conf_emb.pkl"
    item_path = Path(args.item_conformity) if args.item_conformity else root / "item_conf_emb.pkl"
    user_conf = load_embedding(user_path, mat.shape[0], "user_conformity")
    item_conf = load_embedding(item_path, mat.shape[1], "item_conformity")
    scalar = load_or_compute_scalar(args, mat, user_comm)
    texts = write_summaries(summary_path, mat, user_comm, scalar, args.top_items)

    if args.openai_model:
        emb = embed_with_openai(texts, args.openai_model, args.openai_batch_size, args.max_retries)
    else:
        emb = build_default_embeddings(
            mat,
            user_comm,
            user_conf,
            item_conf,
            scalar,
            args.user_weight,
            args.item_weight,
            append_stats=not args.no_append_stats,
        )
    save_pickle(emb.astype(np.float32), output_path)
    print(f"[DONE] Saved community conformity embeddings: {output_path} shape={emb.shape}")


if __name__ == "__main__":
    main()
