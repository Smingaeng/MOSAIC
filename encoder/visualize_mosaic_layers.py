"""Visualize Mosaic layer-wise expert embedding distributions."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "encoder").is_dir() and (parent / "data").is_dir():
            return parent
    return current.parents[1]


PROJECT_ROOT = find_project_root()
ENCODER_DIR = PROJECT_ROOT / "encoder"
DEFAULT_OUTPUT_DIR = ENCODER_DIR / "results" / "mosaic_layer_distributions"

EXPERT_LABELS = {
    0: ("GI", "node intent"),
    1: ("GC", "node conformity"),
    2: ("LI", "community intent product"),
    3: ("LG", "community conformity product"),
}
EXPERT_COLORS = {
    0: "#2563eb",
    1: "#dc2626",
    2: "#059669",
    3: "#d97706",
}
LAYER_COLORS = ["#334155", "#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize mosaic expert distributions across LightGCN layers."
    )
    parser.add_argument("--dataset", default="yelp")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample_size", type=int, default=800)
    parser.add_argument(
        "--scope",
        choices=["selected", "all"],
        default="selected",
        help="Use gate-selected nodes for each expert, or all user/item nodes.",
    )
    parser.add_argument(
        "--embedding_method",
        choices=["tsne", "pca", "both"],
        default="tsne",
        help="2D projection used for point-cloud plots.",
    )
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_max_iter", type=int, default=1000)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def configure_repo_imports(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(ENCODER_DIR))
    sys.argv = [
        sys.argv[0],
        "--model",
        "mosaic",
        "--dataset",
        args.dataset,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()
    return (
        ENCODER_DIR
        / "checkpoint"
        / "mosaic"
        / f"mosaic-{args.dataset}-{args.seed}.pth"
    )


def to_numpy(x):
    return x.detach().float().cpu().numpy()


def summarize_array(values: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
        }
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p25": float(np.percentile(flat, 25)),
        "p50": float(np.percentile(flat, 50)),
        "p75": float(np.percentile(flat, 75)),
        "p95": float(np.percentile(flat, 95)),
    }


def summarize_rows(x: np.ndarray) -> Dict[str, float]:
    norms = np.linalg.norm(np.asarray(x, dtype=np.float32), axis=1)
    return summarize_array(norms)


def cosine_to_reference(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(reference, axis=1)
    return np.sum(x * reference, axis=1) / np.maximum(denom, 1e-12)


def sample_indices(indices: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= sample_size:
        return indices
    return np.sort(rng.choice(indices, size=int(sample_size), replace=False))


def compute_selected_rows(model, scope: str) -> Dict[int, Dict]:
    import torch

    device = model._embedding_device()
    total_nodes = int(model.user_num + model.item_num)
    if scope == "all":
        all_rows = np.arange(total_nodes, dtype=np.int64)
        return {
            int(global_idx): {
                "rows": all_rows,
                "selected_count": int(all_rows.size),
                "fallback_all_nodes": False,
            }
            for global_idx in model.all_expert_ids
        }

    with torch.no_grad():
        w_user, _, _, _ = model.gate(model._gate_features("user", device), model.top_k)
        w_item, _, _, _ = model.gate(model._gate_features("item", device), model.top_k)

    selected_rows: Dict[int, Dict] = {}
    all_rows = np.arange(total_nodes, dtype=np.int64)
    for local_idx, global_idx in enumerate(model.all_expert_ids):
        user_mask = to_numpy(w_user[:, local_idx] > 0).astype(bool)
        item_mask = to_numpy(w_item[:, local_idx] > 0).astype(bool)
        user_rows = np.flatnonzero(user_mask)
        item_rows = np.flatnonzero(item_mask) + int(model.user_num)
        rows = np.concatenate([user_rows, item_rows]).astype(np.int64)
        fallback = rows.size == 0
        selected_rows[int(global_idx)] = {
            "rows": all_rows if fallback else rows,
            "selected_count": int(rows.size),
            "fallback_all_nodes": bool(fallback),
        }
    return selected_rows


def trace_layer_distributions(model, selected_rows: Dict[int, Dict], sample_size: int, seed: int):
    import torch

    rng = np.random.default_rng(seed)
    device = model._embedding_device()
    traces: Dict[int, Dict] = {}

    with torch.no_grad():
        for global_idx in model.all_expert_ids:
            global_idx = int(global_idx)
            local_idx = model.global_to_local_expert[global_idx]
            expert = model.lightgcn_experts[local_idx]
            row_info = selected_rows[global_idx]
            rows_np = sample_indices(row_info["rows"], sample_size, rng)
            rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)

            user_prior = model._raw_node_prior_table("user", global_idx, device=device)
            item_prior = model._raw_node_prior_table("item", global_idx, device=device)
            prior_state = torch.cat([user_prior, item_prior], dim=0)
            embeds = torch.cat([expert.user_embeds, expert.item_embeds], dim=0)
            prop_adj = model.adj.to(device)

            expert_trace = {
                "global_idx": global_idx,
                "local_idx": int(local_idx),
                "sample_rows": rows_np.tolist(),
                "selected_count": int(row_info["selected_count"]),
                "fallback_all_nodes": bool(row_info["fallback_all_nodes"]),
                "layers": [],
            }

            for layer_idx in range(model.layer_num + 1):
                prior_state, gate = expert.layer_modulators[layer_idx](prior_state)
                modulated = gate * embeds
                layer = {
                    "layer": int(layer_idx),
                    "prior_state": to_numpy(prior_state.index_select(0, rows)),
                    "gate": to_numpy(gate.index_select(0, rows)),
                    "pre_embed": to_numpy(embeds.index_select(0, rows)),
                    "modulated": to_numpy(modulated.index_select(0, rows)),
                }
                expert_trace["layers"].append(layer)
                if layer_idx < model.layer_num:
                    embeds = torch.sparse.mm(prop_adj, modulated)

            traces[global_idx] = expert_trace
    return traces


def build_summary(model, traces: Dict[int, Dict], scope: str) -> Dict:
    layer_alpha = to_numpy(model.layer_logits.softmax(dim=-1))
    total_nodes = int(model.user_num + model.item_num)
    summary = {
        "dataset": str(model.dataset),
        "scope": scope,
        "user_num": int(model.user_num),
        "item_num": int(model.item_num),
        "total_nodes": total_nodes,
        "layer_num": int(model.layer_num),
        "top_k": int(model.top_k),
        "active_expert_ids": [int(x) for x in model.all_expert_ids],
        "experts": {},
    }
    for global_idx, trace in traces.items():
        local_idx = int(trace["local_idx"])
        label, description = EXPERT_LABELS.get(global_idx, (f"E{global_idx}", "expert"))
        expert_summary = {
            "label": label,
            "description": description,
            "selected_count": int(trace["selected_count"]),
            "selected_fraction": float(trace["selected_count"] / max(total_nodes, 1)),
            "fallback_all_nodes": bool(trace["fallback_all_nodes"]),
            "sample_count": int(len(trace["sample_rows"])),
            "layer_alpha": [float(x) for x in layer_alpha[local_idx].tolist()],
            "layers": [],
        }
        reference = trace["layers"][0]["modulated"]
        for layer in trace["layers"]:
            drift = cosine_to_reference(layer["modulated"], reference)
            expert_summary["layers"].append(
                {
                    "layer": int(layer["layer"]),
                    "prior_norm": summarize_rows(layer["prior_state"]),
                    "gate": summarize_array(layer["gate"]),
                    "pre_embed_norm": summarize_rows(layer["pre_embed"]),
                    "modulated_norm": summarize_rows(layer["modulated"]),
                    "cosine_to_layer0": summarize_array(drift),
                }
            )
        summary["experts"][str(global_idx)] = expert_summary
    return summary


def fit_pca(points: Iterable[np.ndarray]) -> Tuple[np.ndarray, object]:
    from sklearn.decomposition import PCA

    all_points = np.vstack([np.asarray(x, dtype=np.float32) for x in points])
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(all_points)
    return coords, pca


def fit_tsne(
    points: Iterable[np.ndarray],
    seed: int,
    perplexity: float,
    max_iter: int,
) -> Tuple[np.ndarray, object]:
    from sklearn.manifold import TSNE

    all_points = np.vstack([np.asarray(x, dtype=np.float32) for x in points])
    if all_points.shape[0] < 4:
        raise ValueError("t-SNE needs at least 4 sampled points.")
    perplexity = min(float(perplexity), max(1.0, float(all_points.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    coords = tsne.fit_transform(all_points)
    return coords, tsne


def fit_projection(
    points: Iterable[np.ndarray],
    method: str,
    seed: int,
    perplexity: float,
    max_iter: int,
) -> Tuple[np.ndarray, object]:
    method = str(method).lower()
    if method == "pca":
        return fit_pca(points)
    if method == "tsne":
        return fit_tsne(points, seed, perplexity, max_iter)
    raise ValueError(f"Unknown projection method: {method}")


def plot_layer_projection(
    traces: Dict[int, Dict],
    tensor_key: str,
    output_path: Path,
    dpi: int,
    method: str,
    seed: int,
    perplexity: float,
    max_iter: int,
) -> None:
    plt = ensure_matplotlib()
    arrays: List[np.ndarray] = []
    keys: List[Tuple[int, int, int, int]] = []
    cursor = 0
    for global_idx in sorted(traces):
        for layer in traces[global_idx]["layers"]:
            arr = layer[tensor_key]
            arrays.append(arr)
            next_cursor = cursor + arr.shape[0]
            keys.append((global_idx, int(layer["layer"]), cursor, next_cursor))
            cursor = next_cursor
    coords, projection = fit_projection(arrays, method, seed, perplexity, max_iter)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.reshape(-1)
    axis_prefix = "PC" if method == "pca" else "t-SNE"
    for ax, global_idx in zip(axes, sorted(traces)):
        label, description = EXPERT_LABELS.get(global_idx, (f"E{global_idx}", "expert"))
        ax.set_title(f"{label}: {description}", fontsize=12)
        for key_global_idx, layer_idx, start, end in keys:
            if key_global_idx != global_idx:
                continue
            layer_coords = coords[start:end]
            color = LAYER_COLORS[layer_idx % len(LAYER_COLORS)]
            ax.scatter(
                layer_coords[:, 0],
                layer_coords[:, 1],
                s=4,
                alpha=0.28,
                linewidths=0,
                color=color,
                label=f"L{layer_idx}",
            )
        ax.axhline(0, color="#e5e7eb", lw=0.8)
        ax.axvline(0, color="#e5e7eb", lw=0.8)
        ax.set_xlabel(f"{axis_prefix} 1")
        ax.set_ylabel(f"{axis_prefix} 2")
        ax.legend(markerscale=3, fontsize=9, frameon=False)
    tensor_title = "modulated expert embeddings" if tensor_key == "modulated" else "LI/LG/GI/GC prior states"
    if method == "pca":
        explained = projection.explained_variance_ratio_
        subtitle = f"shared PCA ({explained[0]:.1%}, {explained[1]:.1%})"
    else:
        subtitle = f"shared t-SNE (perplexity={projection.perplexity:.1f})"
    fig.suptitle(f"Layer-wise {tensor_title}, {subtitle}", fontsize=14)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_norms(summary: Dict, output_path: Path, dpi: int) -> None:
    plt = ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    panels = [
        ("prior_norm", "Prior state L2 norm"),
        ("modulated_norm", "Modulated embedding L2 norm"),
    ]
    layers = np.arange(int(summary["layer_num"]) + 1)
    for ax, (metric_name, title) in zip(axes, panels):
        for global_idx_text, expert_summary in summary["experts"].items():
            global_idx = int(global_idx_text)
            label = expert_summary["label"]
            means = np.array([x[metric_name]["mean"] for x in expert_summary["layers"]], dtype=np.float32)
            p25 = np.array([x[metric_name]["p25"] for x in expert_summary["layers"]], dtype=np.float32)
            p75 = np.array([x[metric_name]["p75"] for x in expert_summary["layers"]], dtype=np.float32)
            color = EXPERT_COLORS.get(global_idx, "#111827")
            ax.plot(layers, means, marker="o", color=color, label=label)
            ax.fill_between(layers, p25, p75, color=color, alpha=0.13, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("L2 norm")
        ax.set_xticks(layers)
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.legend(frameon=False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_gate_and_drift(summary: Dict, output_path: Path, dpi: int) -> None:
    plt = ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    layers = np.arange(int(summary["layer_num"]) + 1)
    for global_idx_text, expert_summary in summary["experts"].items():
        global_idx = int(global_idx_text)
        label = expert_summary["label"]
        color = EXPERT_COLORS.get(global_idx, "#111827")

        gate_mean = np.array([x["gate"]["mean"] for x in expert_summary["layers"]], dtype=np.float32)
        gate_p05 = np.array([x["gate"]["p05"] for x in expert_summary["layers"]], dtype=np.float32)
        gate_p95 = np.array([x["gate"]["p95"] for x in expert_summary["layers"]], dtype=np.float32)
        axes[0].plot(layers, gate_mean, marker="o", color=color, label=label)
        axes[0].fill_between(layers, gate_p05, gate_p95, color=color, alpha=0.11, linewidth=0)

        drift_mean = np.array([x["cosine_to_layer0"]["mean"] for x in expert_summary["layers"]], dtype=np.float32)
        drift_p25 = np.array([x["cosine_to_layer0"]["p25"] for x in expert_summary["layers"]], dtype=np.float32)
        drift_p75 = np.array([x["cosine_to_layer0"]["p75"] for x in expert_summary["layers"]], dtype=np.float32)
        axes[1].plot(layers, drift_mean, marker="o", color=color, label=label)
        axes[1].fill_between(layers, drift_p25, drift_p75, color=color, alpha=0.13, linewidth=0)

    axes[0].set_title("Layer modulation gate")
    axes[0].set_ylabel("Gate value")
    axes[1].set_title("Cosine drift from layer 0")
    axes[1].set_ylabel("cos(E_l, E_0)")
    for ax in axes:
        ax.set_xlabel("Layer")
        ax.set_xticks(layers)
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.legend(frameon=False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_report(summary: Dict, out_dir: Path, image_paths: Dict[str, Path]) -> Path:
    report_path = out_dir / "report.md"
    rel_images = {name: path.name for name, path in image_paths.items()}
    lines = [
        f"# Mosaic Layer Distributions: {summary['dataset']}",
        "",
        f"- scope: {summary['scope']}",
        f"- top_k: {summary['top_k']}",
        f"- layers: 0..{summary['layer_num']}",
        f"- nodes: users={summary['user_num']}, items={summary['item_num']}",
        "",
        "## Expert Map",
        "",
        "| Expert id | Label | Description | Selected nodes | Selected fraction | Layer readout alpha |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for global_idx_text, expert in summary["experts"].items():
        alpha = ", ".join(f"{x:.3f}" for x in expert["layer_alpha"])
        selected_note = str(expert["selected_count"])
        if expert["fallback_all_nodes"]:
            selected_note += " (plotted all-node fallback)"
        lines.append(
            f"| {global_idx_text} | {expert['label']} | {expert['description']} | "
            f"{selected_note} | {expert['selected_fraction']:.3f} | {alpha} |"
        )

    lines.extend(
        [
            "",
            "## Visualizations",
            "",
        ]
    )
    for method, display_name in (("tsne", "t-SNE"), ("pca", "PCA")):
        modulated_key = f"modulated_{method}"
        prior_key = f"prior_{method}"
        if modulated_key in rel_images and prior_key in rel_images:
            lines.extend(
                [
                    f"![Modulated embedding {display_name}]({rel_images[modulated_key]})",
                    "",
                    f"![Prior state {display_name}]({rel_images[prior_key]})",
                    "",
                ]
            )
    lines.extend(
        [
            f"![Layer norms]({rel_images['norms']})",
            "",
            f"![Gate and drift]({rel_images['gate_drift']})",
            "",
            "## Layer Summary",
            "",
            "| Expert | Layer | prior norm mean | gate mean | modulated norm mean | cosine to L0 mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for expert in summary["experts"].values():
        for layer in expert["layers"]:
            lines.append(
                f"| {expert['label']} | {layer['layer']} | "
                f"{layer['prior_norm']['mean']:.4f} | "
                f"{layer['gate']['mean']:.4f} | "
                f"{layer['modulated_norm']['mean']:.4f} | "
                f"{layer['cosine_to_layer0']['mean']:.4f} |"
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_repo_imports(args)

    import torch

    from config.configurator import configs
    from data_utils.build_data_handler import build_data_handler
    from models.bulid_model import build_model

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")

    checkpoint = resolve_checkpoint(args)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    data_handler = build_data_handler()
    data_handler.load_data()
    model = build_model(data_handler).to(configs["device"])
    state = torch.load(checkpoint, map_location=configs["device"])
    model.load_state_dict(state, strict=True)
    model.eval()

    selected_rows = compute_selected_rows(model, args.scope)
    traces = trace_layer_distributions(model, selected_rows, args.sample_size, args.seed)
    summary = build_summary(model, traces, args.scope)
    methods = ["tsne", "pca"] if args.embedding_method == "both" else [args.embedding_method]
    summary["projection_methods"] = methods
    if "tsne" in methods:
        summary["tsne"] = {
            "perplexity": float(args.tsne_perplexity),
            "max_iter": int(args.tsne_max_iter),
        }

    out_dir = Path(args.output_dir).expanduser().resolve() / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    image_paths = {
        "norms": out_dir / "layer_norms.png",
        "gate_drift": out_dir / "gate_and_drift.png",
    }
    for method in methods:
        modulated_key = f"modulated_{method}"
        prior_key = f"prior_{method}"
        image_paths[modulated_key] = out_dir / f"modulated_embedding_{method}.png"
        image_paths[prior_key] = out_dir / f"prior_state_{method}.png"
        plot_layer_projection(
            traces,
            "modulated",
            image_paths[modulated_key],
            args.dpi,
            method,
            args.seed,
            args.tsne_perplexity,
            args.tsne_max_iter,
        )
        plot_layer_projection(
            traces,
            "prior_state",
            image_paths[prior_key],
            args.dpi,
            method,
            args.seed,
            args.tsne_perplexity,
            args.tsne_max_iter,
        )
    plot_norms(summary, image_paths["norms"], args.dpi)
    plot_gate_and_drift(summary, image_paths["gate_drift"], args.dpi)
    report_path = write_report(summary, out_dir, image_paths)

    print(f"Wrote report: {report_path}")
    print(f"Wrote summary: {summary_path}")
    for name, path in image_paths.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
