from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import torch as t
from torch import nn
import torch.nn.functional as F

from config.configurator import configs
from models.base_model import BaseModel
from models.loss_utils import cal_infonce_loss, reg_params
from models.model_utils import SpAdjEdgeDrop


init = nn.init.xavier_uniform_
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PriorCache = Dict[Tuple[str, int, str], t.Tensor]


# ---------------------------------------------------------------------------
# Data / feature helpers
# ---------------------------------------------------------------------------


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _as_csr_binary(mat) -> sp.csr_matrix:
    if not sp.issparse(mat):
        mat = sp.csr_matrix(mat)
    mat = mat.tocsr()
    return (mat != 0).astype(np.float32)


def _load_train_matrix(data_handler, data_dir: Path) -> sp.csr_matrix:
    """Use only trn_mat.pkl / data_handler.trn_mat as the observed graph."""
    if hasattr(data_handler, "trn_mat"):
        return _as_csr_binary(data_handler.trn_mat)

    trn_path = data_dir / "trn_mat.pkl"
    if not trn_path.exists():
        raise FileNotFoundError(f"Missing training interaction matrix: {trn_path}")
    return _as_csr_binary(_load_pickle(trn_path))


def _load_community_assignments(
    data_dir: Path,
    dataset: str,
    user_num: int,
    item_num: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    cache_path = data_dir / "community_assignments.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing community assignments for {dataset}: {cache_path}. "
            "Mosaic requires user_community and item_community precomputed in this file."
        )

    cached = _load_pickle(cache_path)
    if not isinstance(cached, dict):
        raise ValueError(f"Community assignment cache must be a dict: {cache_path}")

    required = {"user_community", "item_community"}
    missing = sorted(required - set(cached.keys()))
    if missing:
        raise ValueError(f"Community assignment cache is missing keys {missing}: {cache_path}")

    user_comm = np.asarray(cached["user_community"], dtype=np.int64).reshape(-1)
    item_comm = np.asarray(cached["item_community"], dtype=np.int64).reshape(-1)
    if user_comm.shape[0] != user_num or item_comm.shape[0] != item_num:
        raise ValueError(
            f"Community assignment shape mismatch for {dataset}: "
            f"user/item lengths={(user_comm.shape[0], item_comm.shape[0])}, "
            f"expected={(user_num, item_num)}, path={cache_path}"
        )
    if user_comm.size and int(user_comm.min()) < 0:
        raise ValueError(f"user_community contains negative labels in {cache_path}")
    if item_comm.size and int(item_comm.min()) < 0:
        raise ValueError(f"item_community contains negative labels in {cache_path}")

    inferred = int(max(user_comm.max(initial=0), item_comm.max(initial=0))) + 1
    declared = int(cached.get("num_communities") or inferred)
    if declared < inferred:
        raise ValueError(
            f"num_communities={declared} is smaller than max community label "
            f"{inferred - 1} in {cache_path}"
        )

    metadata = {
        **(cached.get("metadata") or {}),
        "dataset": cached.get("dataset", dataset),
        "num_communities": declared,
        "source": str(cache_path),
    }
    return user_comm, item_comm, metadata


def _safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numer, dtype=np.float32)
    np.divide(numer, denom, out=out, where=denom > 0)
    return out.astype(np.float32)


def _combined_zscore(user_features: np.ndarray, item_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    all_features = np.vstack([user_features, item_features]).astype(np.float32)
    mean = all_features.mean(axis=0, keepdims=True)
    std = all_features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    user_norm = (user_features - mean) / std
    item_norm = (item_features - mean) / std
    return user_norm.astype(np.float32), item_norm.astype(np.float32)


def _participation_from_neighbor_communities(
    node_ids: np.ndarray,
    neighbor_comms: np.ndarray,
    node_count: int,
    degree: np.ndarray,
) -> np.ndarray:
    """Participation coefficient: 1 - sum_c (k_ic / k_i)^2."""
    buckets = [dict() for _ in range(node_count)]
    for node, comm in zip(node_ids, neighbor_comms):
        node = int(node)
        comm = int(comm)
        buckets[node][comm] = buckets[node].get(comm, 0) + 1

    participation = np.zeros(node_count, dtype=np.float32)
    for node, counts in enumerate(buckets):
        if degree[node] <= 0 or not counts:
            continue
        probs = np.asarray(list(counts.values()), dtype=np.float32) / float(degree[node])
        participation[node] = 1.0 - float(np.square(probs).sum())
    return participation


def _build_gate_features(
    trn_mat: sp.csr_matrix,
    user_comm: np.ndarray,
    item_comm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build 4-dim user/item gate inputs for the single shared gate.

    User features:
      1) degree
      2) popularity bias: mean popularity of interacted items
      3) within-community interaction ratio
      4) participation across item communities

    Item features:
      1) popularity
      2) audience activity: mean activity degree of interacting users
      3) within-community interaction ratio
      4) participation across user communities
    """
    trn_mat = _as_csr_binary(trn_mat)
    coo = trn_mat.tocoo()
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)

    user_degree = np.asarray(trn_mat.sum(axis=1)).reshape(-1).astype(np.float32)
    item_popularity = np.asarray(trn_mat.sum(axis=0)).reshape(-1).astype(np.float32)

    # User popularity bias: average item popularity over observed interactions.
    user_pop_sum = np.zeros(trn_mat.shape[0], dtype=np.float32)
    np.add.at(user_pop_sum, rows, item_popularity[cols])
    user_pop_bias = _safe_divide(user_pop_sum, user_degree)

    # Item audience activity: average user degree over observed audience.
    item_audience_sum = np.zeros(trn_mat.shape[1], dtype=np.float32)
    np.add.at(item_audience_sum, cols, user_degree[rows])
    item_audience_activity = _safe_divide(item_audience_sum, item_popularity)

    same_community = (user_comm[rows] == item_comm[cols]).astype(np.float32)
    user_same = np.zeros(trn_mat.shape[0], dtype=np.float32)
    item_same = np.zeros(trn_mat.shape[1], dtype=np.float32)
    np.add.at(user_same, rows, same_community)
    np.add.at(item_same, cols, same_community)
    user_within_ratio = _safe_divide(user_same, user_degree)
    item_within_ratio = _safe_divide(item_same, item_popularity)

    user_participation = _participation_from_neighbor_communities(
        rows,
        item_comm[cols],
        trn_mat.shape[0],
        user_degree,
    )
    item_participation = _participation_from_neighbor_communities(
        cols,
        user_comm[rows],
        trn_mat.shape[1],
        item_popularity,
    )

    user_features = np.stack(
        [
            np.log1p(user_degree),
            np.log1p(user_pop_bias),
            user_within_ratio,
            user_participation,
        ],
        axis=1,
    )
    item_features = np.stack(
        [
            np.log1p(item_popularity),
            np.log1p(item_audience_activity),
            item_within_ratio,
            item_participation,
        ],
        axis=1,
    )
    return _combined_zscore(user_features, item_features)


def _to_numpy_float_array(value, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing {name}; expected a rank-2 embedding matrix.")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 embedding matrix, got shape={arr.shape}")
    return arr


def _prepare_community_embedding(
    arr: np.ndarray,
    num_communities: int,
    raw_dim: int,
    name: str,
) -> np.ndarray:
    if arr.shape[1] != raw_dim:
        raise ValueError(
            f"{name} dimension ({arr.shape[1]}) must match node embedding dimension ({raw_dim})."
        )
    if arr.shape[0] > num_communities:
        raise ValueError(
            f"{name} row count ({arr.shape[0]}) must not exceed num_communities ({num_communities})."
        )
    if arr.shape[0] < num_communities:
        warnings.warn(
            f"{name} has {arr.shape[0]} rows but {num_communities} communities are required; padding zeros."
        )
        pad = np.zeros((num_communities - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return arr.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class PriorProjector(nn.Module):
    """Project a node/community LLM prior into the LightGCN embedding space."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = int(hidden_dim or max(output_dim, (input_dim + output_dim) // 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                init(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x.float())


class LayerWisePriorModulator(nn.Module):
    """Prior-conditioned gate used immediately before one LightGCN propagation hop."""

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        dropout: float,
        max_scale: float,
        lambda_init: float,
        init_std: float,
    ):
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.max_scale = float(max_scale)
        self.init_std = float(init_std)
        if self.max_scale < 0:
            raise ValueError("layer_modulation_scale must be non-negative.")
        lambda_init = float(lambda_init)
        if lambda_init < 0:
            raise ValueError("layer_modulation_lambda_init must be non-negative.")
        if self.max_scale == 0.0:
            lambda_init = 0.0
        elif lambda_init > self.max_scale:
            warnings.warn(
                "layer_modulation_lambda_init is larger than layer_modulation_scale; "
                "clamping the initial layer-wise modulation lambda to the scale cap."
            )
            lambda_init = self.max_scale
        self.lambda_init = lambda_init
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), self.embedding_size),
        )
        self.norm = nn.LayerNorm(self.embedding_size)
        self.lambda_param = nn.Parameter(t.tensor(self.lambda_init, dtype=t.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linears = [module for module in self.mlp if isinstance(module, nn.Linear)]
        for layer in linears:
            if self.init_std > 0:
                nn.init.normal_(layer.weight, mean=0.0, std=self.init_std)
            else:
                nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def modulation_lambda(self) -> t.Tensor:
        return t.clamp(self.lambda_param, min=0.0, max=self.max_scale)

    def forward(self, prior: t.Tensor) -> t.Tensor:
        if prior.shape[-1] != self.embedding_size:
            raise ValueError(
                f"LayerWisePriorModulator expected last dim {self.embedding_size}, got {prior.shape}."
            )
        delta = self.norm(self.mlp(prior.float()))
        lambda_l = self.modulation_lambda().to(device=delta.device, dtype=delta.dtype)
        return 1.0 + lambda_l * t.tanh(delta)


class SharedSoftmaxTopKGate(nn.Module):
    """One shared softmax router used by both user and item nodes."""

    def __init__(self, input_dim: int, hidden_dim: int, expert_num: int):
        super().__init__()
        self.expert_num = int(expert_num)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, expert_num),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.gate:
            if isinstance(module, nn.Linear):
                init(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: t.Tensor, topk: int) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        logits = self.gate(features.float())
        probs = F.softmax(logits, dim=-1)

        topk = max(1, min(int(topk), self.expert_num))
        top_probs, top_idx = t.topk(probs, k=topk, dim=-1)
        top_weights = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-12)

        sparse_weights = t.zeros_like(probs)
        sparse_weights.scatter_(1, top_idx, top_weights)

        selected = (sparse_weights > 0).float()
        importance = probs.mean(dim=0)
        load = sparse_weights.mean(dim=0) if topk == self.expert_num else selected.mean(dim=0)
        balance_loss = self.expert_num * t.sum(importance * load)
        return sparse_weights, top_idx, probs, balance_loss


class LightGCNExpert(nn.Module):
    """Layer-wise modulated LightGCN unit for one MoE expert.

    The expert uses learned user/item node parameters as its initial LightGCN
    embedding. At every layer it computes a prior-conditioned gate and applies
    it immediately before graph propagation, so priors modulate the graph
    filter without replacing the learned E^(0). The final readout is a
    learnable weighted sum of the modulated hop representations.
    """

    def __init__(
        self,
        user_num: int,
        item_num: int,
        embedding_size: int,
        layer_num: int,
        modulation_hidden_size: int,
        modulation_dropout: float,
        modulation_scale: float,
        modulation_lambda_init: float,
        modulation_init_std: float,
    ):
        super().__init__()
        self.user_num = int(user_num)
        self.item_num = int(item_num)
        self.embedding_size = int(embedding_size)
        self.layer_num = int(layer_num)
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.layer_modulators = nn.ModuleList(
            [
                LayerWisePriorModulator(
                    self.embedding_size,
                    modulation_hidden_size,
                    modulation_dropout,
                    modulation_scale,
                    modulation_lambda_init,
                    modulation_init_std,
                )
                for _ in range(self.layer_num + 1)
            ]
        )
        # Each hop has its own module; this assertion catches accidental module reuse.
        assert len({id(module) for module in self.layer_modulators}) == self.layer_num + 1
        self.edge_dropper = SpAdjEdgeDrop()

    def forward(
        self,
        adj: t.Tensor,
        keep_rate: float,
        user_prior: t.Tensor,
        item_prior: t.Tensor,
        layer_logits: t.Tensor,
    ) -> Tuple[t.Tensor, t.Tensor]:
        expected_user = (self.user_num, self.embedding_size)
        expected_item = (self.item_num, self.embedding_size)
        for name, tensor, shape in (
            ("user_prior", user_prior, expected_user),
            ("item_prior", item_prior, expected_item),
        ):
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}.")

        user_embeds = self.user_embeds
        item_embeds = self.item_embeds
        embed_device = user_embeds.device
        if item_embeds.device != embed_device:
            item_embeds = item_embeds.to(device=embed_device, non_blocking=True)
        if user_prior.device != embed_device:
            user_prior = user_prior.to(device=embed_device, non_blocking=True)
        if item_prior.device != embed_device:
            item_prior = item_prior.to(device=embed_device, non_blocking=True)
        if adj.device != embed_device:
            adj = adj.to(device=embed_device)
        if layer_logits.device != embed_device:
            layer_logits = layer_logits.to(device=embed_device, non_blocking=True)
        if layer_logits.numel() != self.layer_num + 1:
            raise ValueError(
                f"layer_logits length must be layer_num + 1 ({self.layer_num + 1}), got {layer_logits.numel()}."
            )

        all_embeds = t.cat([user_embeds, item_embeds], dim=0)
        all_priors = t.cat([user_prior, item_prior], dim=0)
        readout_terms = []
        prop_adj = self.edge_dropper(adj, keep_rate) if self.training else adj
        if prop_adj.device != embed_device:
            prop_adj = prop_adj.to(device=embed_device)

        for layer_idx in range(self.layer_num + 1):
            gate = self.layer_modulators[layer_idx](all_priors)
            if gate.shape != all_embeds.shape:
                raise ValueError(f"Layer-wise gate shape {gate.shape} must match embeddings {all_embeds.shape}.")
            modulated_embeds = gate * all_embeds
            readout_terms.append(modulated_embeds)
            if layer_idx < self.layer_num:
                all_embeds = t.spmm(prop_adj, modulated_embeds)

        assert len(readout_terms) == self.layer_num + 1
        alpha = F.softmax(layer_logits.reshape(-1), dim=0)
        all_embeds = sum(alpha[layer_idx] * readout_terms[layer_idx] for layer_idx in range(self.layer_num + 1))

        return all_embeds[: self.user_num], all_embeds[self.user_num :]


# ---------------------------------------------------------------------------
# Mosaic recommender
# ---------------------------------------------------------------------------


class Mosaic(BaseModel):
    """Mosaic recommender with four routed, prior-modulated LightGCN experts."""

    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.dataset = configs["data"]["name"]
        self.data_dir = PROJECT_ROOT / "data" / self.dataset
        self.expert_num = 4
        self.top_k = int(
            self.hyper_config.get("top_k", configs["model"].get("top_k", 2))
        )
        if not 1 <= self.top_k <= self.expert_num:
            raise ValueError(f"Mosaic flat gate requires top_k in [1, {self.expert_num}].")

        self.intent_expert_ids = (0, 2)
        self.all_expert_ids = tuple(range(self.expert_num))

        self.keep_rate = float(configs["model"].get("keep_rate", 1.0))
        self.layer_num = int(self.hyper_config.get("layer_num", configs["model"].get("layer_num", 3)))
        self.reg_weight = float(self.hyper_config.get("reg_weight", configs["model"].get("reg_weight", 1e-6)))
        self.balance_loss_weight = float(
            self.hyper_config.get("balance_loss_weight", configs["model"].get("balance_loss_weight", 0.0))
        )
        trn_mat = _load_train_matrix(data_handler, self.data_dir)
        user_comm, item_comm, community_meta = _load_community_assignments(
            self.data_dir,
            self.dataset,
            self.user_num,
            self.item_num,
        )
        self.num_communities = int(community_meta["num_communities"])

        self.llm_embedding_device = str(configs["model"].get("llm_embedding_device", "model")).lower()
        if self.llm_embedding_device in {"cuda", "gpu"}:
            self.llm_embedding_device = "model"
        if self.llm_embedding_device not in {"cpu", "model"}:
            raise ValueError("Mosaic llm_embedding_device must be either 'cpu' or 'model'.")

        self.register_buffer("user_community", t.from_numpy(user_comm.astype(np.int64)), persistent=False)
        self.register_buffer("item_community", t.from_numpy(item_comm.astype(np.int64)), persistent=False)

        usrint = _to_numpy_float_array(configs.get("usrint_embeds"), "usrint_embeds")
        itmint = _to_numpy_float_array(configs.get("itmint_embeds"), "itmint_embeds")
        comm_intent = _to_numpy_float_array(configs.get("commint_embeds"), "commint_embeds")
        usrconf = _to_numpy_float_array(configs.get("usrconf_embeds"), "usrconf_embeds")
        itmconf = _to_numpy_float_array(configs.get("itmconf_embeds"), "itmconf_embeds")
        comm_conformity = _to_numpy_float_array(configs.get("commconf_embeds"), "commconf_embeds")
        if usrint.shape[0] != self.user_num or itmint.shape[0] != self.item_num:
            raise ValueError(
                f"Intent embedding shape mismatch: user/item={(usrint.shape[0], itmint.shape[0])}, "
                f"expected={(self.user_num, self.item_num)}"
            )
        if usrconf.shape[0] != self.user_num or itmconf.shape[0] != self.item_num:
            raise ValueError(
                f"Conformity embedding shape mismatch: user/item={(usrconf.shape[0], itmconf.shape[0])}, "
                f"expected={(self.user_num, self.item_num)}"
            )
        if usrint.shape[1] != itmint.shape[1]:
            raise ValueError("user_intent and item_intent dimensions must match for Mosaic priors.")
        if usrconf.shape[1] != itmconf.shape[1]:
            raise ValueError("user_conformity and item_conformity dimensions must match for Mosaic priors.")

        intent_raw_dim = int(usrint.shape[1])
        conformity_raw_dim = int(usrconf.shape[1])
        comm_intent = _prepare_community_embedding(
            comm_intent,
            self.num_communities,
            intent_raw_dim,
            "commint_embeds",
        )
        comm_conformity = _prepare_community_embedding(
            comm_conformity,
            self.num_communities,
            conformity_raw_dim,
            "commconf_embeds",
        )

        self._store_static_feature("user_intent", usrint)
        self._store_static_feature("item_intent", itmint)
        self._store_static_feature("user_conformity", usrconf)
        self._store_static_feature("item_conformity", itmconf)
        self.register_buffer("community_intent", t.from_numpy(comm_intent.astype(np.float32)), persistent=False)
        self.register_buffer(
            "community_conformity",
            t.from_numpy(comm_conformity.astype(np.float32)),
            persistent=False,
        )
        gate_features = self._prepare_gate_features(trn_mat, user_comm, item_comm)
        user_gate_features, item_gate_features = gate_features
        self.gate_input_dim = int(user_gate_features.shape[1])
        self.register_buffer("user_gate_features", t.from_numpy(user_gate_features), persistent=False)
        self.register_buffer("item_gate_features", t.from_numpy(item_gate_features), persistent=False)

        gate_hidden = int(
            self.hyper_config.get(
                "gate_hidden_size",
                configs["model"].get("gate_hidden_size", 64),
            )
        )
        self.gate = SharedSoftmaxTopKGate(
            input_dim=self.gate_input_dim,
            hidden_dim=gate_hidden,
            expert_num=self.expert_num,
        )

        self.prior_projectors = nn.ModuleList(
            [
                PriorProjector(intent_raw_dim, self.embedding_size),
                PriorProjector(conformity_raw_dim, self.embedding_size),
                PriorProjector(intent_raw_dim, self.embedding_size),
                PriorProjector(conformity_raw_dim, self.embedding_size),
            ]
        )

        modulation_hidden = 4 * self.embedding_size
        modulation_dropout = 0.0
        self.layer_modulation_scale = float(
            self.hyper_config.get(
                "layer_modulation_scale",
                configs["model"].get("layer_modulation_scale", 0.5),
            )
        )
        modulation_lambda_init = float(
            self.hyper_config.get(
                "layer_modulation_lambda_init",
                configs["model"].get("layer_modulation_lambda_init", min(self.layer_modulation_scale, 1.0e-2)),
            )
        )
        modulation_init_std = float(
            self.hyper_config.get(
                "layer_modulation_init_std",
                configs["model"].get("layer_modulation_init_std", 1.0e-3),
            )
        )
        self.lightgcn_experts = nn.ModuleList(
            [
                LightGCNExpert(
                    self.user_num,
                    self.item_num,
                    self.embedding_size,
                    self.layer_num,
                    modulation_hidden,
                    modulation_dropout,
                    self.layer_modulation_scale,
                    modulation_lambda_init,
                    modulation_init_std,
                )
                for _ in range(self.expert_num)
            ]
        )
        self.layer_logits = nn.Parameter(t.zeros(self.expert_num, self.layer_num + 1))
        assert self.layer_logits.shape == (self.expert_num, self.layer_num + 1)
        self.llm_align_temperature = float(
            self.hyper_config.get("llm_align_temperature", configs["model"].get("llm_align_temperature", 0.2))
        )
        self.intent_align_weight = float(
            self.hyper_config.get(
                "intent_align_weight",
                configs["model"].get("intent_align_weight", 0.0),
            )
        )

        self._cached_e_user = None
        self._cached_e_item = None

    def _store_static_feature(self, name: str, value: np.ndarray) -> None:
        tensor = t.from_numpy(np.ascontiguousarray(value.astype(np.float32, copy=False)))
        if self.llm_embedding_device == "model":
            self.register_buffer(name, tensor, persistent=False)
        else:
            setattr(self, name, tensor)

    def _feature_table(self, name: str, device: t.device) -> t.Tensor:
        feature = getattr(self, name)
        if feature.device == device:
            return feature
        return feature.to(device=device, non_blocking=True)

    def _prepare_gate_features(
        self,
        trn_mat: sp.csr_matrix,
        user_comm: np.ndarray,
        item_comm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _build_gate_features(trn_mat, user_comm, item_comm)

    def _community_prior_input(self, node_prior: t.Tensor, community_prior: t.Tensor) -> t.Tensor:
        return node_prior * community_prior

    def _embedding_device(self) -> t.device:
        return self.lightgcn_experts[0].user_embeds.device

    def _raw_node_prior_table(
        self,
        node_type: str,
        expert_idx: int,
        device: t.device | None = None,
        prior_cache: PriorCache | None = None,
    ) -> t.Tensor:
        expert_idx = int(expert_idx)
        if not 0 <= expert_idx < self.expert_num:
            raise ValueError(f"Selected expert index out of range: {expert_idx}")
        if device is None:
            device = self._embedding_device()
        device = t.device(device)
        cache_key = (node_type, expert_idx, str(device))
        if prior_cache is not None and cache_key in prior_cache:
            return prior_cache[cache_key]

        if node_type not in {"user", "item"}:
            raise ValueError(f"Unknown node_type: {node_type}")

        if node_type == "user":
            community_ids = self.user_community
            intent_table = self._feature_table("user_intent", device)
            conformity_table = self._feature_table("user_conformity", device)
        else:
            community_ids = self.item_community
            intent_table = self._feature_table("item_intent", device)
            conformity_table = self._feature_table("item_conformity", device)

        if community_ids.device != device:
            community_ids = community_ids.to(device=device, non_blocking=True)
        community_intent = self.community_intent
        community_conformity = self.community_conformity
        if community_intent.device != device:
            community_intent = community_intent.to(device=device, non_blocking=True)
        if community_conformity.device != device:
            community_conformity = community_conformity.to(device=device, non_blocking=True)

        if expert_idx == 0:
            prior_input = intent_table
        elif expert_idx == 1:
            prior_input = conformity_table
        elif expert_idx == 2:
            prior_input = self._community_prior_input(
                intent_table,
                community_intent.index_select(0, community_ids),
            )
        elif expert_idx == 3:
            prior_input = self._community_prior_input(
                conformity_table,
                community_conformity.index_select(0, community_ids),
            )
        prior = self.prior_projectors[expert_idx](prior_input)
        if prior_cache is not None:
            prior_cache[cache_key] = prior
        return prior

    def _gate_features(self, node_type: str, device: t.device | None = None) -> t.Tensor:
        if node_type not in {"user", "item"}:
            raise ValueError(f"Unknown node_type: {node_type}")
        features = self.user_gate_features if node_type == "user" else self.item_gate_features
        if device is not None and features.device != device:
            features = features.to(device=device, non_blocking=True)
        return features

    def _compute_expert_outputs(
        self,
        keep_rate: float,
        prior_cache: PriorCache | None = None,
    ) -> Tuple[Dict[int, t.Tensor], Dict[int, t.Tensor]]:
        device = self._embedding_device()
        expert_ids = self.all_expert_ids

        user_outputs = {}
        item_outputs = {}
        for global_idx in expert_ids:
            global_idx = int(global_idx)
            user_prior = self._raw_node_prior_table(
                "user",
                global_idx,
                device=device,
                prior_cache=prior_cache,
            )
            item_prior = self._raw_node_prior_table(
                "item",
                global_idx,
                device=device,
                prior_cache=prior_cache,
            )
            user_out, item_out = self.lightgcn_experts[global_idx](
                self.adj,
                keep_rate,
                user_prior=user_prior,
                item_prior=item_prior,
                layer_logits=self.layer_logits[global_idx],
            )
            user_outputs[global_idx] = user_out
            item_outputs[global_idx] = item_out
        return user_outputs, item_outputs

    def _route_expert_outputs(
        self,
        expert_outputs: Dict[int, t.Tensor],
        weights: t.Tensor,
        expert_ids: Tuple[int, ...],
        node_ids: t.Tensor | None = None,
    ) -> t.Tensor:
        first_output = expert_outputs[int(expert_ids[0])]
        if node_ids is not None:
            node_ids = node_ids.long()
            if node_ids.device != first_output.device:
                node_ids = node_ids.to(device=first_output.device, non_blocking=True)
            expected_rows = node_ids.shape[0]
        else:
            expected_rows = first_output.shape[0]
        if weights.device != first_output.device:
            weights = weights.to(device=first_output.device, non_blocking=True)
        if weights.shape[0] != expected_rows:
            raise ValueError(f"weights rows {weights.shape[0]} must match routed rows {expected_rows}.")

        routed = first_output.new_zeros((expected_rows, self.embedding_size))
        for local_idx, global_idx in enumerate(expert_ids):
            expert_output = expert_outputs[int(global_idx)]
            if node_ids is not None:
                expert_output = expert_output.index_select(0, node_ids)
            routed = routed + weights[:, local_idx : local_idx + 1] * expert_output
        return routed

    def _clear_embedding_cache(self) -> None:
        self._cached_e_user = None
        self._cached_e_item = None

    def _compute_all_embeddings(self, keep_rate: float, prior_cache: PriorCache | None = None):
        if self.training:
            self._clear_embedding_cache()
        if prior_cache is None:
            prior_cache = {}

        user_outputs, item_outputs = self._compute_expert_outputs(
            keep_rate,
            prior_cache=prior_cache,
        )
        device = user_outputs[int(self.all_expert_ids[0])].device
        w_user, _, _, b_user = self.gate(self._gate_features("user", device), self.top_k)
        w_item, _, _, b_item = self.gate(self._gate_features("item", device), self.top_k)
        final_user = self._route_expert_outputs(
            user_outputs,
            w_user,
            self.all_expert_ids,
        )
        final_item = self._route_expert_outputs(
            item_outputs,
            w_item,
            self.all_expert_ids,
        )

        balance = 0.5 * (b_user + b_item)
        return final_user, final_item, balance

    def _compute_training_batch_embeddings(
        self,
        keep_rate: float,
        ancs: t.Tensor,
        poss: t.Tensor,
        negs: t.Tensor,
        prior_cache: PriorCache | None = None,
    ):
        self._clear_embedding_cache()
        if prior_cache is None:
            prior_cache = {}

        user_outputs, item_outputs = self._compute_expert_outputs(
            keep_rate,
            prior_cache=prior_cache,
        )
        device = user_outputs[int(self.all_expert_ids[0])].device
        w_user, _, _, b_user = self.gate(self._gate_features("user", device), self.top_k)
        w_item, _, _, b_item = self.gate(self._gate_features("item", device), self.top_k)

        batch_users, user_inverse = t.unique(ancs, sorted=False, return_inverse=True)
        batch_item_ids = t.cat([poss, negs], dim=0)
        batch_items, item_inverse = t.unique(batch_item_ids, sorted=False, return_inverse=True)
        pos_inverse = item_inverse[: poss.shape[0]]
        neg_inverse = item_inverse[poss.shape[0] :]

        final_users = self._route_expert_outputs(
            user_outputs,
            w_user.index_select(0, batch_users),
            self.all_expert_ids,
            node_ids=batch_users,
        )
        final_items = self._route_expert_outputs(
            item_outputs,
            w_item.index_select(0, batch_items),
            self.all_expert_ids,
            node_ids=batch_items,
        )

        balance = 0.5 * (b_user + b_item)
        return (
            final_users.index_select(0, user_inverse),
            final_items.index_select(0, pos_inverse),
            final_items.index_select(0, neg_inverse),
            balance,
        )

    def _get_final_embeddings(self, keep_rate: float) -> Tuple[t.Tensor, t.Tensor]:
        if self.training:
            user_embeds, item_embeds, _ = self._compute_all_embeddings(keep_rate)
            return user_embeds, item_embeds
        if self._cached_e_user is None or self._cached_e_item is None:
            user_embeds, item_embeds, _ = self._compute_all_embeddings(keep_rate)
            self._cached_e_user = user_embeds
            self._cached_e_item = item_embeds
        return self._cached_e_user, self._cached_e_item

    def _score_pairs(
        self,
        users: t.Tensor,
        items: t.Tensor,
        return_balance: bool = False,
    ):
        if return_balance:
            all_user_embeds, all_item_embeds, balance = self._compute_all_embeddings(self.keep_rate)
        else:
            all_user_embeds, all_item_embeds = self._get_final_embeddings(self.keep_rate)
            balance = None
        users = users.long()
        items = items.long()
        if users.device != all_user_embeds.device:
            users = users.to(device=all_user_embeds.device, non_blocking=True)
        if items.device != all_item_embeds.device:
            items = items.to(device=all_item_embeds.device, non_blocking=True)
        user_embeds = all_user_embeds.index_select(0, users)
        item_embeds = all_item_embeds.index_select(0, items)
        scores = (user_embeds * item_embeds).sum(dim=-1)

        if return_balance:
            return scores, balance
        return scores

    def forward(self, users: t.Tensor, items: t.Tensor) -> t.Tensor:
        return self._score_pairs(users.long(), items.long(), return_balance=False)

    def cal_loss(self, batch_data):
        ancs, poss, negs = batch_data[:3]
        ancs = ancs.long()
        poss = poss.long()
        negs = negs.long()

        device = self._embedding_device()
        if ancs.device != device:
            ancs = ancs.to(device=device, non_blocking=True)
        if poss.device != device:
            poss = poss.to(device=device, non_blocking=True)
        if negs.device != device:
            negs = negs.to(device=device, non_blocking=True)

        prior_cache: PriorCache = {}
        (
            user_embeds,
            pos_item_embeds,
            neg_item_embeds,
            balance,
        ) = self._compute_training_batch_embeddings(
            self.keep_rate,
            ancs,
            poss,
            negs,
            prior_cache=prior_cache,
        )
        pos_scores = (user_embeds * pos_item_embeds).sum(dim=-1)
        neg_scores = (user_embeds * neg_item_embeds).sum(dim=-1)

        bpr_loss = F.softplus(neg_scores - pos_scores).sum() / ancs.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        balance_loss = self.balance_loss_weight * balance
        llm_align_loss = self._llm_alignment_loss(
            ancs,
            poss,
            negs,
            user_embeds,
            pos_item_embeds,
            neg_item_embeds,
            prior_cache=prior_cache,
        )
        loss = bpr_loss + reg_loss + balance_loss + llm_align_loss
        losses = {
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
            "balance_loss": balance_loss,
        }
        if self.intent_align_weight > 0:
            losses["llm_align_loss"] = llm_align_loss
        return loss, losses

    def _llm_alignment_loss(
        self,
        ancs: t.Tensor,
        poss: t.Tensor,
        negs: t.Tensor,
        final_user_embeds: t.Tensor,
        final_pos_item_embeds: t.Tensor,
        final_neg_item_embeds: t.Tensor,
        prior_cache: PriorCache | None = None,
    ) -> t.Tensor:
        if self.intent_align_weight <= 0:
            return final_user_embeds.new_tensor(0.0)

        loss = final_user_embeds.new_tensor(0.0)
        for expert_idx in self.intent_expert_ids:
            loss = loss + self._expert_alignment_loss(
                ancs,
                poss,
                negs,
                None,
                None,
                expert_idx,
                prior_cache=prior_cache,
                batch_user_embeds=final_user_embeds,
                batch_pos_item_embeds=final_pos_item_embeds,
                batch_neg_item_embeds=final_neg_item_embeds,
            )
        return self.intent_align_weight * loss / len(self.intent_expert_ids)

    def _expert_alignment_loss(
        self,
        ancs: t.Tensor,
        poss: t.Tensor,
        negs: t.Tensor,
        user_embeds_all: t.Tensor | None,
        item_embeds_all: t.Tensor | None,
        expert_idx: int,
        prior_cache: PriorCache | None = None,
        batch_user_embeds: t.Tensor | None = None,
        batch_pos_item_embeds: t.Tensor | None = None,
        batch_neg_item_embeds: t.Tensor | None = None,
    ) -> t.Tensor:
        batch_size = max(int(ancs.shape[0]), 1)
        temp = self.llm_align_temperature
        loss_ref = user_embeds_all if user_embeds_all is not None else batch_user_embeds
        if loss_ref is None:
            raise ValueError("Either full or batch user embeddings are required for expert alignment loss.")
        user_prior = self._raw_node_prior_table(
            "user",
            expert_idx,
            device=loss_ref.device,
            prior_cache=prior_cache,
        )
        item_prior = self._raw_node_prior_table(
            "item",
            expert_idx,
            device=loss_ref.device,
            prior_cache=prior_cache,
        )
        if batch_user_embeds is None:
            if user_embeds_all is None or item_embeds_all is None:
                raise ValueError("Full user/item embeddings are required when batch embeddings are not provided.")
            user_embeds = user_embeds_all.index_select(0, ancs)
            pos_item_embeds = item_embeds_all.index_select(0, poss)
            neg_item_embeds = item_embeds_all.index_select(0, negs)
        else:
            if batch_pos_item_embeds is None or batch_neg_item_embeds is None:
                raise ValueError("Batch positive/negative item embeddings are required with batch user embeddings.")
            user_embeds = batch_user_embeds
            pos_item_embeds = batch_pos_item_embeds
            neg_item_embeds = batch_neg_item_embeds
        return (
            cal_infonce_loss(user_embeds, user_prior.index_select(0, ancs), user_prior, temp)
            + cal_infonce_loss(pos_item_embeds, item_prior.index_select(0, poss), item_prior, temp)
            + cal_infonce_loss(neg_item_embeds, item_prior.index_select(0, negs), item_prior, temp)
        ) / batch_size

    @t.no_grad()
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        user_all, item_embeds = self._get_final_embeddings(1.0)
        if pck_users.device != user_all.device:
            pck_users = pck_users.to(device=user_all.device, non_blocking=True)
        user_embeds = user_all.index_select(0, pck_users)
        full_preds = user_embeds @ item_embeds.T
        if train_mask.device != full_preds.device:
            train_mask = train_mask.to(device=full_preds.device, non_blocking=True)
        return self._mask_predict(full_preds, train_mask)
