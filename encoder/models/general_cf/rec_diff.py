"""RecDiff adapted to CoMoE's general_cf interface.

This file ports the uploaded RecDiff implementation into the CoMoE model API:

* ``forward`` builds user-item GCN embeddings and user-user/social GCN embeddings.
* ``cal_loss`` optimizes BPR + L2 regularization + hidden-space diffusion loss.
* ``full_predict`` adds the reverse-diffused social representation to the
  collaborative user embedding before all-rank scoring.

The original RecDiff code depends on DGL.  CoMoE already represents the
user-item graph as a normalized ``torch.sparse`` adjacency, so this port keeps
RecDiff's message-passing equation but implements it with PyTorch sparse matrix
multiplication.

For non-social ``amazon_*`` datasets, this adapter can derive and cache
``data/<dataset>/user_edges.pkl`` from ``community_assignments.pkl`` before
falling back to generic missing-social-graph behavior.
"""

from __future__ import annotations

import math
import pickle
import warnings
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch as t
from torch import nn
import torch.nn.functional as F

from config.configurator import configs
from models.base_model import BaseModel


init = nn.init.xavier_uniform_
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _coalesce_sparse_tensor(adj: t.Tensor) -> t.Tensor:
    return adj.coalesce() if adj.is_sparse else adj


def _scipy_to_torch_sparse(mat: sp.spmatrix, device: t.device) -> t.Tensor:
    mat = mat.tocoo().astype(np.float32)
    idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = t.from_numpy(mat.data.astype(np.float32))
    return t.sparse_coo_tensor(idxs, vals, mat.shape, device=device).coalesce()


def _recdiff_message_norm(mat: sp.spmatrix) -> sp.coo_matrix:
    """Return the sparse matrix used by RecDiff-style message passing.

    The source RecDiff DGL layer normalizes source node features by out-degree,
    sends them through directed edges ``src -> dst``, then normalizes the
    resulting destination features by in-degree.  For ``torch.sparse.mm`` this
    is ``D_in^{-1/2} A^T D_out^{-1/2}`` with rows as receivers and columns as
    senders.
    """

    mat = (mat.tocsr() != 0).astype(np.float32)
    out_degree = np.asarray(mat.sum(axis=1)).reshape(-1).astype(np.float32)
    in_degree = np.asarray(mat.sum(axis=0)).reshape(-1).astype(np.float32)

    out_inv_sqrt = np.power(np.maximum(out_degree, 1.0), -0.5)
    in_inv_sqrt = np.power(np.maximum(in_degree, 1.0), -0.5)

    return sp.diags(in_inv_sqrt).dot(mat.transpose()).dot(sp.diags(out_inv_sqrt)).tocoo()


def _to_user_graph_matrix(obj: Any, user_num: int) -> Optional[sp.coo_matrix]:
    """Best-effort conversion of common social-graph file formats to scipy COO."""

    if sp.issparse(obj):
        mat = obj.tocoo()
        if mat.shape != (user_num, user_num):
            raise ValueError(f"social graph shape {mat.shape} does not match ({user_num}, {user_num})")
        return mat

    if isinstance(obj, dict):
        # RecDiff datasets store the social matrix under 'trust'.  Other CoMoE
        # preprocessing scripts often use edge-oriented names.
        for key in (
            "trust",
            "social",
            "social_mat",
            "uu_mat",
            "user_graph",
            "uu_graph",
            "adj",
            "matrix",
        ):
            if key in obj:
                return _to_user_graph_matrix(obj[key], user_num)

        if "edge_index" in obj:
            edge_index = np.asarray(obj["edge_index"])
            if edge_index.ndim == 2 and edge_index.shape[0] == 2:
                rows, cols = edge_index[0], edge_index[1]
            elif edge_index.ndim == 2 and edge_index.shape[1] == 2:
                rows, cols = edge_index[:, 0], edge_index[:, 1]
            else:
                raise ValueError(f"Unsupported social edge_index shape: {edge_index.shape}")
            return _edges_to_user_graph(rows, cols, user_num)

        row_key = "row" if "row" in obj else "src" if "src" in obj else None
        col_key = "col" if "col" in obj else "dst" if "dst" in obj else None
        if row_key is not None and col_key is not None:
            return _edges_to_user_graph(obj[row_key], obj[col_key], user_num)

    arr = np.asarray(obj)
    if arr.ndim == 2 and arr.shape == (user_num, user_num):
        return sp.coo_matrix(arr)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return _edges_to_user_graph(arr[:, 0], arr[:, 1], user_num)
    if arr.ndim == 2 and arr.shape[0] == 2:
        return _edges_to_user_graph(arr[0], arr[1], user_num)

    return None


def _edges_to_user_graph(rows: Iterable[Any], cols: Iterable[Any], user_num: int) -> sp.coo_matrix:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(cols, dtype=np.int64).reshape(-1)
    valid = (rows >= 0) & (rows < user_num) & (cols >= 0) & (cols < user_num)
    rows = rows[valid]
    cols = cols[valid]
    data = np.ones(rows.shape[0], dtype=np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(user_num, user_num))


# ---------------------------------------------------------------------------
# RecDiff modules
# ---------------------------------------------------------------------------


class RecDiffGCNLayer(nn.Module):
    """User-item GCN layer from RecDiff, implemented with sparse tensors."""

    def __init__(self, embedding_size: int, use_weight: bool = True, activation: Optional[nn.Module] = None):
        super().__init__()
        self.use_weight = bool(use_weight)
        self.activation = activation
        if self.use_weight:
            self.u_w = nn.Parameter(t.empty(embedding_size, embedding_size))
            self.v_w = nn.Parameter(t.empty(embedding_size, embedding_size))
            init(self.u_w)
            init(self.v_w)

    def forward(self, adj: t.Tensor, user_embeds: t.Tensor, item_embeds: t.Tensor) -> t.Tensor:
        if self.use_weight:
            user_embeds = user_embeds @ self.u_w
            item_embeds = item_embeds @ self.v_w
        node_embeds = t.cat([user_embeds, item_embeds], dim=0)
        out = t.sparse.mm(adj, node_embeds)
        if self.activation is not None:
            out = self.activation(out)
        return out


class RecDiffUUGCNLayer(nn.Module):
    """User-user social GCN layer from RecDiff."""

    def __init__(self, embedding_size: int, use_weight: bool = True, activation: Optional[nn.Module] = None):
        super().__init__()
        self.use_weight = bool(use_weight)
        self.activation = activation
        if self.use_weight:
            self.u_w = nn.Parameter(t.empty(embedding_size, embedding_size))
            init(self.u_w)

    def forward(self, adj: t.Tensor, user_embeds: t.Tensor) -> t.Tensor:
        if self.use_weight:
            user_embeds = user_embeds @ self.u_w
        out = t.sparse.mm(adj, user_embeds)
        if self.activation is not None:
            out = self.activation(out)
        return out


class SDNet(nn.Module):
    """The RecDiff denoising network for the reverse diffusion process."""

    def __init__(
        self,
        in_dims: Iterable[int],
        out_dims: Iterable[int],
        emb_size: int,
        time_type: str = "cat",
        norm: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_dims = list(in_dims)
        self.out_dims = list(out_dims)
        if self.out_dims[0] != self.in_dims[-1]:
            raise ValueError("SDNet requires out_dims[0] == in_dims[-1], matching the RecDiff source.")
        self.time_type = time_type
        self.time_emb_dim = int(emb_size)
        self.norm = bool(norm)
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type != "cat":
            raise ValueError(f"Unimplemented timestep embedding type: {self.time_type}")
        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

        self.in_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])]
        )
        self.out_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.out_dims[:-1], self.out_dims[1:])]
        )
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self) -> None:
        for layer in chain(self.in_layers, self.out_layers, [self.emb_layer]):
            size = layer.weight.size()
            fan_out, fan_in = size[0], size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x: t.Tensor, timesteps: t.Tensor) -> t.Tensor:
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x, p=2, dim=-1)
        x = self.drop(x)
        h = t.cat([x, emb], dim=-1)
        for layer in self.in_layers:
            h = t.tanh(layer(h))
        for idx, layer in enumerate(self.out_layers):
            h = layer(h)
            if idx != len(self.out_layers) - 1:
                h = t.tanh(h)
        return h


class DiffusionProcess(nn.Module):
    """Gaussian diffusion process from RecDiff, with tensors registered as buffers."""

    def __init__(
        self,
        noise_schedule: str,
        noise_scale: float,
        noise_min: float,
        noise_max: float,
        steps: int,
        keep_num: int = 10,
    ):
        super().__init__()
        self.noise_schedule = str(noise_schedule)
        self.noise_scale = float(noise_scale)
        self.noise_min = float(noise_min)
        self.noise_max = float(noise_max)
        self.steps = int(steps)
        self.keep_num = int(keep_num)

        beta_nums = t.tensor(self.betas_num(), dtype=t.float64)
        if len(beta_nums.shape) != 1:
            raise ValueError("betas must be 1-D")
        if len(beta_nums) != self.steps:
            raise ValueError("number of betas must equal diffusion steps")
        if not bool(((beta_nums > 0) & (beta_nums <= 1)).all()):
            raise ValueError("betas out of range")

        self.register_buffer("Lt_record", t.zeros(self.steps, self.keep_num, dtype=t.float64))
        self.register_buffer("Lt_count", t.zeros(self.steps, dtype=t.long))
        self.register_buffer("beta_nums", beta_nums)
        self._register_diffusion_buffers()

    def betas_num(self) -> np.ndarray:
        st_bound = self.noise_scale * self.noise_min
        e_bound = self.noise_scale * self.noise_max
        variance = np.linspace(st_bound, e_bound, self.steps, dtype=np.float64)
        if self.noise_schedule == "linear":
            return variance
        return betas_from_linear_variance(self.steps, variance)

    def _register_or_update_buffer(self, name: str, value: t.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    def _register_diffusion_buffers(self) -> None:
        alphas = 1.0 - self.beta_nums
        alphas_cumprod = t.cumprod(alphas, dim=0)
        alphas_cumprod_prev = t.cat([t.tensor([1.0], dtype=t.float64, device=alphas.device), alphas_cumprod[:-1]])
        alphas_cumprod_next = t.cat([alphas_cumprod[1:], t.tensor([0.0], dtype=t.float64, device=alphas.device)])

        posterior_variance = self.beta_nums * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = t.log(t.cat([posterior_variance[1].unsqueeze(0), posterior_variance[1:]]))
        posterior_mean_coef1 = self.beta_nums * t.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * t.sqrt(alphas) / (1.0 - alphas_cumprod)

        buffers = {
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "alphas_cumprod_next": alphas_cumprod_next,
            "sqrt_alphas_cumprod": t.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": t.sqrt(1.0 - alphas_cumprod),
            "log_one_minus_alphas_cumprod": t.log(1.0 - alphas_cumprod),
            "sqrt_recip_alphas_cumprod": t.sqrt(1.0 / alphas_cumprod),
            "sqrt_recipm1_alphas_cumprod": t.sqrt(1.0 / alphas_cumprod - 1.0),
            "posterior_variance": posterior_variance,
            "posterior_log_variance_clipped": posterior_log_variance_clipped,
            "posterior_mean_coef1": posterior_mean_coef1,
            "posterior_mean_coef2": posterior_mean_coef2,
        }
        for name, value in buffers.items():
            self._register_or_update_buffer(name, value)

    def calculate_losses(self, model: nn.Module, emb_s: t.Tensor, reweight: bool = False) -> Dict[str, t.Tensor]:
        batch_size, device = emb_s.size(0), emb_s.device
        ts, _ = self.sample_timesteps(batch_size, device, method="uniform")
        noise = t.randn_like(emb_s)
        emb_t = self.forward_process(emb_s, ts, noise)
        model_output = model(emb_t, ts)
        if model_output.shape != emb_s.shape:
            raise ValueError(f"SDNet output shape {model_output.shape} does not match input {emb_s.shape}")

        mse = mean_flat((emb_s - model_output) ** 2)
        if reweight:
            weight = self.SNR(ts - 1) - self.SNR(ts)
            weight = t.where(ts == 0, t.ones_like(weight), weight)
        else:
            weight = t.ones_like(mse)

        return {"loss": weight.float() * mse, "pred_xstart": model_output}

    # Keep the source typo as a compatibility alias.
    def caculate_losses(self, model: nn.Module, emb_s: t.Tensor, reweight: bool = False) -> Dict[str, t.Tensor]:
        return self.calculate_losses(model, emb_s, reweight)

    def sample_timesteps(
        self,
        batch_size: int,
        device: t.device,
        method: str = "uniform",
        uniform_prob: float = 0.001,
    ) -> Tuple[t.Tensor, t.Tensor]:
        if method == "importance":
            if not bool((self.Lt_count == self.keep_num).all()):
                return self.sample_timesteps(batch_size, device, method="uniform")
            Lt_sqrt = t.sqrt(t.mean(self.Lt_record ** 2, dim=-1))
            pt_all = Lt_sqrt / t.sum(Lt_sqrt)
            pt_all = pt_all * (1.0 - uniform_prob) + uniform_prob / len(pt_all)
            picked_t = t.multinomial(pt_all, num_samples=batch_size, replacement=True).to(device)
            pt = pt_all.to(device).gather(dim=0, index=picked_t) * len(pt_all)
            return picked_t.long(), pt.float()
        if method == "uniform":
            picked_t = t.randint(0, self.steps, (batch_size,), device=device).long()
            pt = t.ones_like(picked_t).float()
            return picked_t, pt
        raise ValueError(f"Unsupported timestep sampling method: {method}")

    def forward_process(self, emb_s: t.Tensor, timesteps: t.Tensor, noise: Optional[t.Tensor] = None) -> t.Tensor:
        if noise is None:
            noise = t.randn_like(emb_s)
        if noise.shape != emb_s.shape:
            raise ValueError("Diffusion noise must have the same shape as the embedding input.")
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, timesteps, emb_s.shape) * emb_s
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timesteps, emb_s.shape) * noise
        )

    # The uploaded source calls q_sample inside p_sample but only defines the
    # equivalent forward_process.  Provide the alias so nonzero sampling_steps work.
    def q_sample(self, emb_s: t.Tensor, timesteps: t.Tensor, noise: Optional[t.Tensor] = None) -> t.Tensor:
        return self.forward_process(emb_s, timesteps, noise)

    def q_posterior_mean_variance(self, emb_s: t.Tensor, emb_t: t.Tensor, timesteps: t.Tensor):
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, timesteps, emb_t.shape) * emb_s
            + self._extract_into_tensor(self.posterior_mean_coef2, timesteps, emb_t.shape) * emb_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, timesteps, emb_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, timesteps, emb_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model: nn.Module, x: t.Tensor, timesteps: t.Tensor) -> Dict[str, t.Tensor]:
        if timesteps.shape != (x.shape[0],):
            raise ValueError("timesteps must be a 1-D tensor with one entry per batch row")
        model_output = model(x, timesteps)
        model_variance = self._extract_into_tensor(self.posterior_variance, timesteps, x.shape)
        model_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, timesteps, x.shape)
        pred_xstart = model_output
        model_mean, _, _ = self.q_posterior_mean_variance(emb_s=pred_xstart, emb_t=x, timesteps=timesteps)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(self, model: nn.Module, emb_s: t.Tensor, steps: int, sampling_noise: bool = False) -> t.Tensor:
        steps = int(steps)
        if steps > self.steps:
            raise ValueError("Too many steps in inference.")
        if steps == 0:
            emb_t = emb_s
        else:
            timesteps = t.full((emb_s.shape[0],), steps - 1, dtype=t.long, device=emb_s.device)
            emb_t = self.q_sample(emb_s, timesteps)

        indices = list(range(self.steps))[::-1]
        if self.noise_scale == 0.0:
            for idx in indices:
                timesteps = t.full((emb_t.shape[0],), idx, dtype=t.long, device=emb_s.device)
                emb_t = model(emb_t, timesteps)
            return emb_t

        for idx in indices:
            timesteps = t.full((emb_t.shape[0],), idx, dtype=t.long, device=emb_s.device)
            out = self.p_mean_variance(model, emb_t, timesteps)
            if sampling_noise:
                noise = t.randn_like(emb_t)
                nonzero_mask = (timesteps != 0).float().view(-1, *([1] * (len(emb_t.shape) - 1)))
                emb_t = out["mean"] + nonzero_mask * t.exp(0.5 * out["log_variance"]) * noise
            else:
                emb_t = out["mean"]
        return emb_t

    def SNR(self, timesteps: t.Tensor) -> t.Tensor:
        return self.alphas_cumprod[timesteps] / (1.0 - self.alphas_cumprod[timesteps])

    @staticmethod
    def _extract_into_tensor(arr: t.Tensor, timesteps: t.Tensor, broadcast_shape: Tuple[int, ...]) -> t.Tensor:
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


# ---------------------------------------------------------------------------
# CoMoE model wrapper
# ---------------------------------------------------------------------------


class Rec_diff(BaseModel):
    """RecDiff model implemented for CoMoE's pairwise/all-rank trainer."""

    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.data_handler = data_handler
        self.device = t.device(configs.get("device", "cuda"))
        self.adj = _coalesce_sparse_tensor(data_handler.torch_adj)
        self.trn_mat = data_handler.trn_mat

        self.keep_rate = float(self._cfg("keep_rate", 1.0))
        self.layer_num = int(self._cfg("layer_num", self._cfg("n_layers", 2)))
        self.social_layer_num = int(self._cfg("social_layer_num", self._cfg("s_layers", 2)))
        self.use_layer_weight = bool(self._cfg("weight", True))
        self.reg_weight = float(self._cfg("reg_weight", self._cfg("reg", 1e-2)))
        self.diff_loss_weight = float(self._cfg("diff_loss_weight", 1.0))
        self.diff_lr = float(self._cfg("diff_lr", self._cfg("difflr", configs.get("optimizer", {}).get("lr", 1e-3))))
        self.reweight = bool(self._cfg("reweight", True))
        self.sampling_steps = int(self._cfg("sampling_steps", 0))
        self.sampling_noise = bool(self._cfg("sampling_noise", False))

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        act = nn.LeakyReLU(0.5, inplace=False)
        self.ui_layers = nn.ModuleList(
            [RecDiffGCNLayer(self.embedding_size, use_weight=self.use_layer_weight, activation=act) for _ in range(self.layer_num)]
        )
        self.uu_layers = nn.ModuleList(
            [RecDiffUUGCNLayer(self.embedding_size, use_weight=self.use_layer_weight, activation=act) for _ in range(self.social_layer_num)]
        )

        dims = int(self._cfg("dims", self.embedding_size))
        emb_size = int(self._cfg("emb_size", 16))
        norm = bool(self._cfg("norm", True))
        dropout = float(self._cfg("dropout", 0.0))
        output_dims = [dims, self.embedding_size]
        input_dims = output_dims[::-1]
        self.sdnet = SDNet(input_dims, output_dims, emb_size, time_type="cat", norm=norm, dropout=dropout)
        self.diffusion = DiffusionProcess(
            noise_schedule=str(self._cfg("noise_schedule", "linear-var")),
            noise_scale=float(self._cfg("noise_scale", 1.0)),
            noise_min=float(self._cfg("noise_min", 1.0e-4)),
            noise_max=float(self._cfg("noise_max", 1.0e-2)),
            steps=int(self._cfg("steps", 20)),
        )

        self.social_adj = self._build_social_adj().coalesce()
        self.edge_dropper = None  # kept for API/debug symmetry with other CoMoE models
        self.final_ui_embeds: Optional[t.Tensor] = None
        self.final_uu_embeds: Optional[t.Tensor] = None
        self.is_training = False

    def _cfg(self, key: str, default: Any = None) -> Any:
        if isinstance(getattr(self, "hyper_config", None), dict) and key in self.hyper_config:
            return self.hyper_config[key]
        return configs["model"].get(key, default)

    def get_optimizer_param_groups(self, base_lr: float):
        """Use RecDiff's two learning rates inside CoMoE's single optimizer hook."""

        cf_params = list(
            chain(
                [self.user_embeds, self.item_embeds],
                self.ui_layers.parameters(),
                self.uu_layers.parameters(),
            )
        )
        return [
            {"params": cf_params, "lr": float(base_lr)},
            {"params": self.sdnet.parameters(), "lr": self.diff_lr},
        ]

    def _data_dir(self) -> Path:
        return PROJECT_ROOT / "data" / configs["data"]["name"]

    def _resolve_social_path(self, path_like: str) -> Path:
        path = Path(str(path_like))
        if path.is_absolute():
            return path
        return self._data_dir() / path

    def _social_candidates(self):
        explicit = self._cfg("social_graph_path", None)
        if explicit not in (None, "", "null"):
            yield self._resolve_social_path(str(explicit))

        for name in self._cfg(
            "social_graph_candidates",
            [
                "user_edges.pkl",
                "trust.pkl",
                "trust_mat.pkl",
                "social.pkl",
                "social_mat.pkl",
                "social_edges.pkl",
                "uu_graph.pkl",
                "uu_mat.pkl",
                "user_graph.pkl",
            ],
        ):
            yield self._resolve_social_path(str(name))

    def _load_social_matrix(self) -> Tuple[Optional[sp.coo_matrix], Optional[Path]]:
        seen = set()
        for path in self._social_candidates():
            if path in seen:
                continue
            seen.add(path)
            if not path.exists():
                continue
            with path.open("rb") as f:
                obj = pickle.load(f)
            mat = _to_user_graph_matrix(obj, self.user_num)
            if mat is None:
                raise ValueError(f"Could not parse social graph file: {path}")
            return mat, path
        derived = self._maybe_build_amazon_community_user_edges()
        if derived is not None:
            mat, assignment_path, user_edges_path = derived
            warnings.warn(
                "RecDiff built missing amazon_* user-user graph from "
                f"{assignment_path} and cached it at {user_edges_path}.",
                RuntimeWarning,
            )
            return mat, user_edges_path
        return None, None

    def _maybe_build_amazon_community_user_edges(self) -> Optional[Tuple[sp.coo_matrix, Path, Path]]:
        dataset = str(configs.get("data", {}).get("name", ""))
        if not dataset.startswith("amazon_"):
            return None

        data_dir = self._data_dir()
        user_edges_path = data_dir / "user_edges.pkl"
        if user_edges_path.exists():
            return None

        assignment_path = data_dir / "community_assignments.pkl"
        if not assignment_path.exists():
            raise FileNotFoundError(
                f"RecDiff could not find user_edges.pkl for {dataset} and cannot derive it "
                f"because community_assignments.pkl is missing: {assignment_path}"
            )

        user_comm = self._load_user_community_labels(assignment_path)
        social = self._community_labels_to_user_edges(user_comm)
        self._dump_pickle_atomic(user_edges_path, social.tocsr())
        return social.tocoo(), assignment_path, user_edges_path

    def _load_user_community_labels(self, path: Path) -> np.ndarray:
        with path.open("rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            if "user_community" not in obj:
                raise ValueError(f"community_assignments.pkl must contain 'user_community': {path}")
            labels = obj["user_community"]
        else:
            labels = obj

        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels.shape[0] != self.user_num:
            raise ValueError(
                f"community_assignments user length mismatch for {configs['data']['name']}: "
                f"expected={self.user_num}, got={labels.shape[0]}, path={path}"
            )
        if labels.size and int(labels.min()) < 0:
            raise ValueError(f"user_community contains negative labels: {path}")
        return labels

    def _community_labels_to_user_edges(self, labels: np.ndarray) -> sp.coo_matrix:
        if self.user_num == 0:
            return sp.coo_matrix((0, 0), dtype=np.float32)

        _, compact_labels = np.unique(labels, return_inverse=True)
        num_communities = int(compact_labels.max()) + 1 if compact_labels.size else 0
        rows = np.arange(self.user_num, dtype=np.int64)
        membership = sp.csr_matrix(
            (
                np.ones(self.user_num, dtype=np.float32),
                (rows, compact_labels.astype(np.int64, copy=False)),
            ),
            shape=(self.user_num, num_communities),
            dtype=np.float32,
        )
        social = (membership @ membership.T).tocsr().astype(np.float32)
        social.setdiag(0.0)
        social.data = np.ones_like(social.data, dtype=np.float32)
        social.eliminate_zeros()
        return social.tocoo()

    @staticmethod
    def _dump_pickle_atomic(path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)

    def _fallback_social_matrix(self) -> sp.coo_matrix:
        fallback = str(self._cfg("social_graph_fallback", "identity")).lower()
        if fallback in {"identity", "self_loop", "self-loops"}:
            return sp.eye(self.user_num, dtype=np.float32, format="coo")
        if fallback in {"none", "empty"}:
            return sp.coo_matrix((self.user_num, self.user_num), dtype=np.float32)
        if fallback in {"cooccurrence", "co_interaction", "co-interaction"}:
            # Optional fallback for non-social CoMoE datasets.  This is not the
            # RecDiff paper setting, but it preserves the two-branch interface
            # when explicit user-user ties are unavailable.
            mat = self.trn_mat.tocsr().astype(np.float32)
            uu = (mat @ mat.transpose()).tocsr()
            uu.setdiag(0.0)
            uu.eliminate_zeros()
            max_neighbors = int(self._cfg("fallback_max_social_neighbors", 0) or 0)
            if max_neighbors > 0:
                uu = self._prune_topk_rows(uu, max_neighbors)
            return (uu != 0).astype(np.float32).tocoo()
        raise ValueError(
            "model.social_graph_fallback must be one of {'identity', 'none', 'cooccurrence'}, "
            f"got {fallback!r}"
        )

    @staticmethod
    def _prune_topk_rows(mat: sp.csr_matrix, topk: int) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        for row in range(mat.shape[0]):
            start, end = mat.indptr[row], mat.indptr[row + 1]
            row_cols = mat.indices[start:end]
            row_data = mat.data[start:end]
            if row_data.size > topk:
                keep = np.argpartition(row_data, -topk)[-topk:]
                row_cols = row_cols[keep]
                row_data = row_data[keep]
            rows.extend([row] * len(row_cols))
            cols.extend(row_cols.tolist())
            data.extend(row_data.tolist())
        return sp.csr_matrix((data, (rows, cols)), shape=mat.shape, dtype=np.float32)

    def _build_social_adj(self) -> t.Tensor:
        social_mat, social_path = self._load_social_matrix()
        if social_mat is None:
            if bool(self._cfg("require_social_graph", False)):
                raise FileNotFoundError(
                    f"RecDiff requires a user-user social graph, but none of the configured files exist under {self._data_dir()}."
                )
            social_mat = self._fallback_social_matrix()
            warnings.warn(
                "RecDiff social graph file was not found. Falling back to "
                f"model.social_graph_fallback={self._cfg('social_graph_fallback', 'identity')!r}. "
                "For paper-faithful RecDiff experiments, provide a user-user trust graph via model.social_graph_path.",
                RuntimeWarning,
            )
        elif bool(self._cfg("social_graph_undirected", False)):
            social_mat = ((social_mat + social_mat.transpose()) != 0).astype(np.float32).tocoo()

        if bool(self._cfg("add_social_self_loop", False)):
            social_mat = (social_mat + sp.eye(self.user_num, dtype=np.float32, format="coo")).tocoo()

        norm_social = _recdiff_message_norm(social_mat)
        return _scipy_to_torch_sparse(norm_social, self.adj.device)

    def _propagate_ui(self, adj: t.Tensor) -> t.Tensor:
        all_embeddings = [t.cat([self.user_embeds, self.item_embeds], dim=0)]
        embeddings: Optional[t.Tensor] = None
        for layer_idx, layer in enumerate(self.ui_layers):
            if layer_idx == 0:
                embeddings = layer(adj, self.user_embeds, self.item_embeds)
            else:
                assert embeddings is not None
                embeddings = layer(adj, embeddings[: self.user_num], embeddings[self.user_num :])
            all_embeddings.append(F.normalize(embeddings, p=2, dim=1))
        return sum(all_embeddings)

    def _propagate_uu(self, social_adj: t.Tensor) -> t.Tensor:
        all_embeddings = [self.user_embeds]
        embeddings: Optional[t.Tensor] = None
        for layer_idx, layer in enumerate(self.uu_layers):
            if layer_idx == 0:
                embeddings = layer(social_adj, self.user_embeds)
            else:
                assert embeddings is not None
                embeddings = layer(social_adj, embeddings)
            all_embeddings.append(F.normalize(embeddings, p=2, dim=1))
        return sum(all_embeddings)

    def forward(self, adj: Optional[t.Tensor] = None, social_adj: Optional[t.Tensor] = None):
        if adj is None:
            adj = self.adj
        if social_adj is None:
            social_adj = self.social_adj

        if not self.is_training and self.final_ui_embeds is not None and self.final_uu_embeds is not None:
            ui_embeds = self.final_ui_embeds
            uu_embeds = self.final_uu_embeds
        else:
            ui_embeds = self._propagate_ui(adj)
            uu_embeds = self._propagate_uu(social_adj)
            if not self.is_training:
                self.final_ui_embeds = ui_embeds
                self.final_uu_embeds = uu_embeds

        return ui_embeds[: self.user_num], ui_embeds[self.user_num :], uu_embeds

    @staticmethod
    def _pair_score(user_embeds: t.Tensor, item_embeds: t.Tensor) -> t.Tensor:
        return (user_embeds * item_embeds).sum(dim=-1)

    def cal_loss(self, batch_data):
        self.is_training = True
        self.final_ui_embeds = None
        self.final_uu_embeds = None

        ancs, poss, negs = batch_data
        user_embeds, item_embeds, uu_embeds = self.forward(self.adj, self.social_adj)

        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        diff_terms = self.diffusion.calculate_losses(self.sdnet, uu_embeds[ancs], self.reweight)
        diff_loss = diff_terms["loss"].mean()
        enhanced_anc_embeds = anc_embeds + diff_terms["pred_xstart"]

        pos_scores = self._pair_score(enhanced_anc_embeds, pos_embeds)
        neg_scores = self._pair_score(enhanced_anc_embeds, neg_embeds)
        batch_size = max(int(ancs.shape[0]), 1)
        bpr_loss = F.softplus(neg_scores - pos_scores).sum() / batch_size
        reg_loss = (
            enhanced_anc_embeds.norm(2).square() + pos_embeds.norm(2).square() + neg_embeds.norm(2).square()
        ) * self.reg_weight / batch_size

        loss = bpr_loss + reg_loss + self.diff_loss_weight * diff_loss
        losses = {
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
            "diff_loss": diff_loss,
        }
        if self.diff_loss_weight != 1.0:
            losses["weighted_diff_loss"] = self.diff_loss_weight * diff_loss
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        user_embeds, item_embeds, uu_embeds = self.forward(self.adj, self.social_adj)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        pck_user_embeds = user_embeds[pck_users]
        social_embeds = uu_embeds[pck_users]
        user_predict = self.diffusion.p_sample(self.sdnet, social_embeds, self.sampling_steps, self.sampling_noise)
        full_preds = (pck_user_embeds + user_predict) @ item_embeds.T
        return self._mask_predict(full_preds, train_mask)


# Convenient alias for readers/importers; CoMoE's builder instantiates Rec_diff.
RecDiff = Rec_diff


# ---------------------------------------------------------------------------
# Diffusion/math utilities
# ---------------------------------------------------------------------------


def timestep_embedding(timesteps: t.Tensor, dim: int, max_period: int = 10000) -> t.Tensor:
    half = dim // 2
    freqs = t.exp(-math.log(max_period) * t.arange(start=0, end=half, dtype=t.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = t.cat([t.cos(args), t.sin(args)], dim=-1)
    if dim % 2:
        embedding = t.cat([embedding, t.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def mean_flat(tensor: t.Tensor) -> t.Tensor:
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def betas_from_linear_variance(steps: int, variance: np.ndarray, max_beta: float = 0.999) -> np.ndarray:
    alpha_bar = 1 - variance
    betas = [1 - alpha_bar[0]]
    for idx in range(1, steps):
        betas.append(min(1 - alpha_bar[idx] / alpha_bar[idx - 1], max_beta))
    return np.asarray(betas, dtype=np.float64)
