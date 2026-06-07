"""Community detection and feature helpers for CoMoE.

This module intentionally uses only trn_mat.pkl and optional user-user edge files.
For yelp/epnions it prefers Louvain on the user-user graph.  For amazon_book it
uses a coordinate-ascent heuristic for Barber's bipartite modularity objective:

    Q = 1/m * sum_{u,i} (A_ui - k_u d_i / m) * 1[c_u == c_i]

The Barber heuristic alternates user-label and item-label updates under this
objective.  It is deterministic given --seed and avoids validation/test leakage.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

try:
    from .pickle_compat import load_pickle_compat
except ImportError:  # pragma: no cover - direct script execution
    from pickle_compat import load_pickle_compat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOCIAL_GRAPH_CANDIDATES = (
    "social_mat.pkl",
    "trust_mat.pkl",
    "user_user_mat.pkl",
    "uu_mat.pkl",
    "user_mat.pkl",
    "friend_mat.pkl",
    "relation_mat.pkl",
    "net_mat.pkl",
    "social_edges.pkl",
    "trust_edges.pkl",
    "user_edges.pkl",
)


def data_dir_for(dataset: str, data_dir: Optional[str] = None) -> Path:
    return Path(data_dir) if data_dir else PROJECT_ROOT / "data" / dataset


def load_pickle(path: Path):
    return load_pickle_compat(path)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_train_matrix(dataset: str, data_dir: Optional[str] = None) -> sp.csr_matrix:
    path = data_dir_for(dataset, data_dir) / "trn_mat.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing training matrix: {path}")
    mat = load_pickle(path)
    mat = (mat != 0).astype(np.float32)
    if not sp.issparse(mat):
        mat = sp.csr_matrix(mat)
    return mat.tocsr()


def compact_labels(labels: np.ndarray, keep_negative: bool = False) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    out = np.full_like(labels, -1 if keep_negative else 0)
    valid = labels >= 0 if keep_negative else np.ones(labels.shape[0], dtype=bool)
    unique = np.unique(labels[valid])
    mapping = {int(old): new for new, old in enumerate(unique)}
    for old, new in mapping.items():
        out[labels == old] = new
    return out.astype(np.int64)


def load_social_graph(dataset: str, data_dir: Optional[str] = None) -> Tuple[Optional[sp.csr_matrix], Optional[Path]]:
    root = data_dir_for(dataset, data_dir)
    for name in SOCIAL_GRAPH_CANDIDATES:
        path = root / name
        if not path.exists():
            continue
        obj = load_pickle(path)
        if sp.issparse(obj):
            graph = (obj != 0).astype(np.float32).tocsr()
        else:
            arr = np.asarray(obj)
            if arr.ndim == 2:
                graph = sp.csr_matrix((arr != 0).astype(np.float32))
            elif arr.ndim == 1:
                raise ValueError(f"Unsupported 1-D social graph object in {path}")
            else:
                raise ValueError(f"Unsupported social graph object shape in {path}: {arr.shape}")
        # Force square, symmetric, binary graph with no self-loop.
        if graph.shape[0] != graph.shape[1]:
            raise ValueError(f"Social graph must be square: {path} shape={graph.shape}")
        graph = graph.maximum(graph.T).tocsr()
        graph.setdiag(0)
        graph.eliminate_zeros()
        return graph, path
    return None, None


def communities_from_networkx_louvain(graph: sp.csr_matrix, seed: int) -> np.ndarray:
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("networkx is required for Louvain community detection.") from exc

    if hasattr(nx, "from_scipy_sparse_array"):
        g = nx.from_scipy_sparse_array(graph, create_using=nx.Graph)
    else:  # older networkx
        g = nx.from_scipy_sparse_matrix(graph, create_using=nx.Graph)

    try:
        comms = nx.algorithms.community.louvain_communities(g, weight="weight", seed=seed)
    except Exception as exc:
        warnings.warn(f"networkx Louvain failed ({exc}); using connected components fallback.")
        comms = list(nx.connected_components(g))

    labels = np.full(graph.shape[0], -1, dtype=np.int64)
    for cid, nodes in enumerate(comms):
        for node in nodes:
            labels[int(node)] = cid

    # Isolated nodes may be omitted by old community implementations.
    missing = np.where(labels < 0)[0]
    next_cid = int(labels.max()) + 1 if labels.max() >= 0 else 0
    for node in missing:
        labels[node] = next_cid
        next_cid += 1
    return compact_labels(labels)


def choose_barber_k(user_num: int, item_num: int, nonzeros: int, requested: Optional[int] = None) -> int:
    if requested is not None and requested > 0:
        return int(requested)
    # A conservative heuristic: enough communities to be local, not enough to
    # make every local graph tiny.  Works for sparse recommender matrices.
    active_scale = max(2.0, np.sqrt(max(1, nonzeros) / 20.0))
    size_scale = np.sqrt(max(2, user_num + item_num)) / 2.5
    return int(np.clip(round(min(active_scale, size_scale)), 2, 128))


def initialize_barber_labels(mat: sp.csr_matrix, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    user_num, item_num = mat.shape
    user_degree = np.asarray(mat.sum(axis=1)).reshape(-1)
    item_degree = np.asarray(mat.sum(axis=0)).reshape(-1)

    # Degree-aware deterministic bins plus tiny random jitter to break ties.
    def init_by_degree(deg: np.ndarray, n: int) -> np.ndarray:
        labels = np.zeros(n, dtype=np.int64)
        active = np.where(deg > 0)[0]
        if active.size == 0:
            return rng.integers(0, k, size=n, endpoint=False).astype(np.int64)
        order = active[np.argsort(-(deg[active] + 1e-6 * rng.random(active.size)))]
        for rank, idx in enumerate(order):
            labels[idx] = rank % k
        isolated = np.where(deg <= 0)[0]
        labels[isolated] = rng.integers(0, k, size=isolated.size, endpoint=False)
        return labels

    return init_by_degree(user_degree, user_num), init_by_degree(item_degree, item_num)


def barber_modularity(mat: sp.csr_matrix, user_labels: np.ndarray, item_labels: np.ndarray) -> float:
    mat = mat.tocsr()
    m = float(mat.nnz)
    if m <= 0:
        return 0.0
    user_degree = np.asarray(mat.sum(axis=1)).reshape(-1).astype(np.float64)
    item_degree = np.asarray(mat.sum(axis=0)).reshape(-1).astype(np.float64)
    q = 0.0
    coo = mat.tocoo()
    same = user_labels[coo.row] == item_labels[coo.col]
    q += float(same.sum())
    labels = np.union1d(np.unique(user_labels), np.unique(item_labels))
    for label in labels:
        ku = user_degree[user_labels == label].sum()
        di = item_degree[item_labels == label].sum()
        q -= ku * di / m
    return q / m


def barber_bipartite_communities(
    mat: sp.csr_matrix,
    n_clusters: Optional[int] = None,
    max_iter: int = 30,
    seed: int = 2023,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Coordinate-ascent heuristic for Barber bipartite modularity.

    For fixed item labels, each user chooses the label maximizing
        observed_edges_to_label - k_u * total_item_degree_of_label / m.
    For fixed user labels, each item uses the symmetric update.  This directly
    optimizes Barber's bipartite modularity local assignment objective.
    """
    mat = (mat != 0).astype(np.float32).tocsr()
    user_num, item_num = mat.shape
    m = float(mat.nnz)
    if m <= 0:
        raise ValueError("Cannot run Barber community detection on an empty train matrix.")
    k = choose_barber_k(user_num, item_num, mat.nnz, n_clusters)
    user_labels, item_labels = initialize_barber_labels(mat, k, seed)
    user_degree = np.asarray(mat.sum(axis=1)).reshape(-1).astype(np.float64)
    item_degree = np.asarray(mat.sum(axis=0)).reshape(-1).astype(np.float64)
    mat_csc = mat.tocsc()

    last_q = barber_modularity(mat, user_labels, item_labels)
    if verbose:
        print(f"[INFO] Barber init: k={k}, Q={last_q:.6f}")

    for it in range(max_iter):
        item_degree_by_label = np.bincount(item_labels, weights=item_degree, minlength=k).astype(np.float64)
        user_changed = 0
        for u in range(user_num):
            start, end = mat.indptr[u], mat.indptr[u + 1]
            neigh = mat.indices[start:end]
            observed = np.bincount(item_labels[neigh], minlength=k).astype(np.float64) if neigh.size else np.zeros(k)
            scores = observed - user_degree[u] * item_degree_by_label / m
            new_label = int(scores.argmax())
            if new_label != user_labels[u]:
                user_changed += 1
                user_labels[u] = new_label

        user_degree_by_label = np.bincount(user_labels, weights=user_degree, minlength=k).astype(np.float64)
        item_changed = 0
        for i in range(item_num):
            start, end = mat_csc.indptr[i], mat_csc.indptr[i + 1]
            neigh = mat_csc.indices[start:end]
            observed = np.bincount(user_labels[neigh], minlength=k).astype(np.float64) if neigh.size else np.zeros(k)
            scores = observed - item_degree[i] * user_degree_by_label / m
            new_label = int(scores.argmax())
            if new_label != item_labels[i]:
                item_changed += 1
                item_labels[i] = new_label

        q = barber_modularity(mat, user_labels, item_labels)
        if verbose:
            print(f"[INFO] Barber iter={it + 1:02d} Q={q:.6f} user_changed={user_changed} item_changed={item_changed}")
        if user_changed == 0 and item_changed == 0:
            break
        # Stop if modularity is numerically stable.  Small decreases may happen
        # because updates are greedy and labels are shared, so do not rollback.
        if abs(q - last_q) < 1e-8:
            break
        last_q = q

    user_labels = compact_labels(user_labels)
    # Map item labels into the compact user label id space where possible.
    label_map = {old: new for new, old in enumerate(np.unique(user_labels))}
    item_labels = np.asarray([label_map.get(int(x), -1) for x in item_labels], dtype=np.int64)
    item_labels = compact_labels(item_labels, keep_negative=True)
    meta = {
        "method": "barber_coordinate_ascent",
        "requested_clusters": n_clusters,
        "initial_clusters": k,
        "iterations": it + 1,
        "modularity": float(barber_modularity(mat, user_labels, item_labels)),
    }
    return user_labels, item_labels, meta


def fallback_bipartite_for_social_missing(mat: sp.csr_matrix, seed: int, max_iter: int, n_clusters: Optional[int]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    warnings.warn(
        "User-user social graph was not found. Falling back to Barber bipartite communities from trn_mat.pkl."
    )
    u, i, meta = barber_bipartite_communities(mat, n_clusters=n_clusters, max_iter=max_iter, seed=seed, verbose=True)
    meta["method"] = "barber_fallback_no_social_graph"
    return u, i, meta


def detect_communities(
    dataset: str,
    data_dir: Optional[str] = None,
    method: str = "auto",
    seed: int = 2023,
    barber_clusters: Optional[int] = None,
    barber_max_iter: int = 30,
) -> Dict:
    mat = load_train_matrix(dataset, data_dir)
    dataset_lower = dataset.lower()
    chosen = method.lower()
    if chosen == "auto":
        if dataset_lower in {"yelp", "epnions"}:
            chosen = "louvain"
        elif dataset_lower == "amazon_book":
            chosen = "barber"
        else:
            chosen = "barber"

    item_comm = None
    if chosen == "louvain":
        social, social_path = load_social_graph(dataset, data_dir)
        if social is None:
            user_comm, item_comm, meta = fallback_bipartite_for_social_missing(
                mat, seed=seed, max_iter=barber_max_iter, n_clusters=barber_clusters
            )
        else:
            user_comm = communities_from_networkx_louvain(social, seed)
            item_comm = assign_items_to_majority_user_community(mat, user_comm)
            meta = {
                "method": "louvain",
                "social_graph": str(social_path),
                "num_social_edges": int(social.nnz // 2),
            }
    elif chosen == "barber":
        user_comm, item_comm, meta = barber_bipartite_communities(
            mat, n_clusters=barber_clusters, max_iter=barber_max_iter, seed=seed, verbose=True
        )
    else:
        raise ValueError(f"Unsupported community method: {method}. Use auto, louvain, or barber.")

    user_comm = compact_labels(user_comm)
    if item_comm is None:
        item_comm = np.full(mat.shape[1], -1, dtype=np.int64)
    else:
        item_comm = compact_labels(item_comm, keep_negative=True)
    result = {
        "dataset": dataset,
        "user_community": user_comm.astype(np.int64),
        "item_community": item_comm.astype(np.int64),
        "num_communities": int(user_comm.max()) + 1 if user_comm.size else 0,
        "metadata": {
            **meta,
            "seed": seed,
            "train_shape": tuple(mat.shape),
            "train_nnz": int(mat.nnz),
        },
    }
    return result


def assign_items_to_majority_user_community(mat: sp.csr_matrix, user_comm: np.ndarray) -> np.ndarray:
    mat = mat.tocsc()
    out = np.full(mat.shape[1], -1, dtype=np.int64)
    for i in range(mat.shape[1]):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        users = mat.indices[start:end]
        if users.size == 0:
            continue
        labels = user_comm[users]
        values, counts = np.unique(labels, return_counts=True)
        out[i] = int(values[np.argmax(counts)])
    return out


def write_assignment_summary(result: Dict, output_path: Path) -> None:
    csv_path = output_path.with_suffix(".summary.csv")
    user_comm = result["user_community"]
    counts = Counter(user_comm.tolist())
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("community_id,num_users\n")
        for cid, count in sorted(counts.items()):
            f.write(f"{cid},{count}\n")
    print(f"[DONE] Saved community summary: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CoMoE community assignments from training data only.")
    parser.add_argument("--dataset", required=True, help="Dataset name under data/<dataset>.")
    parser.add_argument("--data-dir", default=None, help="Optional explicit data directory.")
    parser.add_argument("--method", default="auto", choices=["auto", "louvain", "barber"], help="Community method.")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--barber-clusters", type=int, default=None, help="Optional number of Barber communities.")
    parser.add_argument("--barber-max-iter", type=int, default=30)
    parser.add_argument("--output", default=None, help="Output pkl path. Default: data/<dataset>/community_assignments.pkl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = data_dir_for(args.dataset, args.data_dir)
    output = Path(args.output) if args.output else root / "community_assignments.pkl"
    if output.exists() and not args.overwrite:
        print(f"[SKIP] {output} exists. Use --overwrite to rebuild.")
        return
    result = detect_communities(
        dataset=args.dataset,
        data_dir=args.data_dir,
        method=args.method,
        seed=args.seed,
        barber_clusters=args.barber_clusters,
        barber_max_iter=args.barber_max_iter,
    )
    save_pickle(result, output)
    print(
        f"[DONE] Saved {output} method={result['metadata'].get('method')} "
        f"num_communities={result['num_communities']}"
    )
    write_assignment_summary(result, output)


if __name__ == "__main__":
    main()
