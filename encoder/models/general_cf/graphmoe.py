from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch as t
from torch import nn
import torch.nn.functional as F

from config.configurator import configs
from models.base_model import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params
from models.model_utils import SpAdjEdgeDrop


init = nn.init.xavier_uniform_


class NoisyTopKGating(nn.Module):
    """Noisy sparse top-k router from Graph-Mixture-of-Experts.

    The original repository applies this router to graph-property/node-property
    PyG mini-batches.  Here it is kept as a dense node-wise router over the full
    user-item bipartite graph so that it can be trained by CoMoE's pairwise BPR
    recommendation loop.
    """

    def __init__(
        self,
        input_size: int,
        num_experts: int,
        k: int,
        noisy_gating: bool = True,
        gate_init: str = "xavier",
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.num_experts = int(num_experts)
        self.k = int(k)
        self.noisy_gating = bool(noisy_gating)

        if self.num_experts < 1:
            raise ValueError("num_experts must be positive.")
        if self.k < 1 or self.k > self.num_experts:
            raise ValueError("k/top_k must be in [1, num_experts].")

        self.w_gate = nn.Parameter(t.empty(self.input_size, self.num_experts))
        self.w_noise = nn.Parameter(t.empty(self.input_size, self.num_experts))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", t.tensor([0.0]))
        self.register_buffer("std", t.tensor([1.0]))
        self.reset_parameters(gate_init=gate_init)

    def reset_parameters(self, gate_init: str = "xavier") -> None:
        gate_init = str(gate_init).lower()
        if gate_init == "zero":
            nn.init.zeros_(self.w_gate)
        else:
            init(self.w_gate)
        nn.init.zeros_(self.w_noise)

    @staticmethod
    def cv_squared(x: t.Tensor) -> t.Tensor:
        """Squared coefficient of variation used as the load-balancing term."""
        if x.numel() <= 1:
            return x.new_tensor(0.0)
        eps = 1e-10
        return x.float().var(unbiased=False) / (x.float().mean().square() + eps)

    @staticmethod
    def _gates_to_load(gates: t.Tensor) -> t.Tensor:
        return (gates > 0).sum(dim=0).float()

    def _prob_in_top_k(
        self,
        clean_values: t.Tensor,
        noisy_values: t.Tensor,
        noise_stddev: t.Tensor,
        noisy_top_values: t.Tensor,
    ) -> t.Tensor:
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = t.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = t.gather(top_values_flat, 0, threshold_positions_if_in).unsqueeze(1)
        is_in = noisy_values > threshold_if_in

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = t.gather(top_values_flat, 0, threshold_positions_if_out).unsqueeze(1)

        normal = t.distributions.Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return t.where(is_in, prob_if_in, prob_if_out)

    def forward(self, x: t.Tensor, training: bool) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        """Return sparse gates, top-k indices, clean logits, and balance loss."""
        clean_logits = x @ self.w_gate
        if self.noisy_gating and training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-2
            noisy_logits = clean_logits + t.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            noise_stddev = None
            noisy_logits = clean_logits
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        gates = t.zeros_like(logits)
        gates.scatter_(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and training:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(dim=0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(dim=0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        return gates, top_k_indices, clean_logits, balance_loss


class GraphMoEExpert(nn.Module):
    """One LightGCN-style graph expert for a user-item bipartite graph."""

    def __init__(
        self,
        embedding_size: int,
        hop: int = 1,
        use_linear: bool = True,
        use_batchnorm: bool = True,
        normalize_output: bool = False,
    ):
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.hop = int(hop)
        self.use_linear = bool(use_linear)
        self.normalize_output = bool(normalize_output)

        if self.hop < 1:
            raise ValueError("Expert hop must be >= 1.")

        self.linear = nn.Linear(self.embedding_size, self.embedding_size, bias=False) if self.use_linear else nn.Identity()
        self.norm = nn.BatchNorm1d(self.embedding_size) if use_batchnorm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if isinstance(self.linear, nn.Linear):
            init(self.linear.weight)

    def forward(self, x: t.Tensor, adj: t.Tensor) -> t.Tensor:
        out = x
        for _ in range(self.hop):
            out = t.spmm(adj, out)
        out = self.linear(out)
        out = self.norm(out)
        if self.normalize_output:
            out = F.normalize(out, p=2, dim=-1)
        return out


class GraphMoELayer(nn.Module):
    """A sparse Graph-MoE message-passing layer for recommendation."""

    def __init__(
        self,
        embedding_size: int,
        num_experts: int,
        top_k: int,
        num_experts_1hop: Optional[int] = None,
        noisy_gating: bool = True,
        gate_init: str = "xavier",
        expert_linear: bool = True,
        expert_batchnorm: bool = True,
        expert_l2norm: bool = False,
        combine: str = "sum",
    ):
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.combine = str(combine).lower()

        if num_experts_1hop is None:
            num_experts_1hop = self.num_experts
        self.num_experts_1hop = int(num_experts_1hop)
        if self.num_experts_1hop < 0 or self.num_experts_1hop > self.num_experts:
            raise ValueError("num_experts_1hop must be in [0, num_experts].")

        self.router = NoisyTopKGating(
            input_size=self.embedding_size,
            num_experts=self.num_experts,
            k=self.top_k,
            noisy_gating=noisy_gating,
            gate_init=gate_init,
        )
        self.experts = nn.ModuleList()
        for expert_idx in range(self.num_experts):
            hop = 1 if expert_idx < self.num_experts_1hop else 2
            self.experts.append(
                GraphMoEExpert(
                    embedding_size=self.embedding_size,
                    hop=hop,
                    use_linear=expert_linear,
                    use_batchnorm=expert_batchnorm,
                    normalize_output=expert_l2norm,
                )
            )

        self._last_gates: Optional[t.Tensor] = None
        self._last_topk: Optional[t.Tensor] = None
        self._last_logits: Optional[t.Tensor] = None

    def forward(self, x: t.Tensor, adj: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        gates, topk_indices, logits, balance_loss = self.router(x, self.training)

        expert_outputs = [expert(x, adj) for expert in self.experts]
        expert_outputs = t.stack(expert_outputs, dim=1)  # [node_num, expert_num, emb_dim]
        out = t.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        if self.combine in {"mean", "legacy_mean"}:
            out = out / max(1, self.num_experts)

        self._last_gates = gates.detach()
        self._last_topk = topk_indices.detach()
        self._last_logits = logits.detach()
        return out, balance_loss


class GraphMoE(BaseModel):
    """Graph-Mixture-of-Experts recommender for CoMoE's general_cf task.

    This model adapts the original Graph-Mixture-of-Experts idea from graph
    property prediction to implicit-feedback recommendation:

    * nodes are users and items in CoMoE's bipartite interaction graph;
    * every layer owns several graph experts;
    * each node is routed to top-k experts by a noisy sparse gate;
    * the output is optimized with CoMoE's pairwise BPR objective.
    """

    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.adj = data_handler.torch_adj.coalesce()
        self.edge_dropper = SpAdjEdgeDrop()

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        # Hyperparameters. Dataset-specific entries override global model keys.
        self.keep_rate = float(self._cfg("keep_rate", 1.0))
        self.layer_num = int(self._cfg("layer_num", 3))
        self.reg_weight = float(self._cfg("reg_weight", 1.0e-6))
        self.num_experts = int(self._cfg("num_experts", 4))
        self.top_k = int(self._cfg("top_k", self._cfg("k", 2)))
        self.num_experts_1hop = self._cfg("num_experts_1hop", self.num_experts)
        if self.num_experts_1hop is not None:
            self.num_experts_1hop = int(self.num_experts_1hop)

        self.noisy_gating = bool(self._cfg("noisy_gating", True))
        self.gate_init = str(self._cfg("gate_init", "xavier"))
        self.balance_loss_weight = float(self._cfg("balance_loss_weight", self._cfg("coef", 1.0e-2)))
        self.dropout = float(self._cfg("dropout", self._cfg("drop_ratio", 0.0)))
        self.residual = bool(self._cfg("residual", False))
        self.jk = str(self._cfg("jk", self._cfg("JK", "sum"))).lower()
        self.activation = str(self._cfg("activation", "relu")).lower()
        self.last_layer_activation = bool(self._cfg("last_layer_activation", False))
        self.expert_linear = bool(self._cfg("expert_linear", True))
        self.expert_batchnorm = bool(self._cfg("expert_batchnorm", True))
        self.expert_l2norm = bool(self._cfg("expert_l2norm", False))
        self.combine = str(self._cfg("combine", "sum"))
        self.log_expert_usage = bool(self._cfg("log_expert_usage", False))

        self.layers = nn.ModuleList(
            [
                GraphMoELayer(
                    embedding_size=self.embedding_size,
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    num_experts_1hop=self.num_experts_1hop,
                    noisy_gating=self.noisy_gating,
                    gate_init=self.gate_init,
                    expert_linear=self.expert_linear,
                    expert_batchnorm=self.expert_batchnorm,
                    expert_l2norm=self.expert_l2norm,
                    combine=self.combine,
                )
                for _ in range(self.layer_num)
            ]
        )

        self.final_embeds: Optional[t.Tensor] = None
        self._last_balance_loss: Optional[t.Tensor] = None

    def _cfg(self, key: str, default=None):
        if isinstance(getattr(self, "hyper_config", None), dict) and key in self.hyper_config:
            return self.hyper_config[key]
        return configs["model"].get(key, default)

    def _activate(self, x: t.Tensor) -> t.Tensor:
        if self.activation in {"none", "identity", "linear"}:
            return x
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "leaky_relu":
            return F.leaky_relu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _propagate(self, adj: t.Tensor, embeds: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        embeds_list: List[t.Tensor] = [embeds]
        balance_terms: List[t.Tensor] = []
        h = embeds

        for layer_idx, layer in enumerate(self.layers):
            h_in = h
            h, balance_loss = layer(h, adj)
            balance_terms.append(balance_loss)

            is_last_layer = layer_idx == self.layer_num - 1
            if (not is_last_layer) or self.last_layer_activation:
                h = self._activate(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual:
                h = h + h_in
            embeds_list.append(h)

        if self.jk == "last":
            final_embeds = embeds_list[-1]
        elif self.jk == "sum":
            final_embeds = sum(embeds_list)
        elif self.jk == "mean":
            final_embeds = sum(embeds_list) / len(embeds_list)
        else:
            raise ValueError("jk must be one of {last, sum, mean}.")

        if balance_terms:
            balance_loss = t.stack(balance_terms).mean()
        else:
            balance_loss = embeds.new_tensor(0.0)
        return final_embeds, balance_loss

    def forward(
        self,
        adj: Optional[t.Tensor] = None,
        keep_rate: Optional[float] = None,
        return_aux: bool = False,
    ):
        if adj is None:
            adj = self.adj
        if keep_rate is None:
            keep_rate = self.keep_rate

        if not self.training and self.final_embeds is not None and not return_aux:
            user_embeds = self.final_embeds[: self.user_num]
            item_embeds = self.final_embeds[self.user_num :]
            return user_embeds, item_embeds

        if self.training:
            self.final_embeds = None
            adj = self.edge_dropper(adj, keep_rate)

        embeds = t.cat([self.user_embeds, self.item_embeds], dim=0)
        final_embeds, balance_loss = self._propagate(adj, embeds)
        self._last_balance_loss = balance_loss.detach()

        if not self.training:
            self.final_embeds = final_embeds.detach() if bool(self._cfg("detach_eval_cache", False)) else final_embeds

        user_embeds = final_embeds[: self.user_num]
        item_embeds = final_embeds[self.user_num :]
        if return_aux:
            return user_embeds, item_embeds, {"balance_loss": balance_loss}
        return user_embeds, item_embeds

    def cal_loss(self, batch_data):
        user_embeds, item_embeds, aux = self.forward(self.adj, self.keep_rate, return_aux=True)
        ancs, poss, negs = batch_data[:3]

        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / max(1, anc_embeds.shape[0])
        reg_loss = self.reg_weight * reg_params(self)
        balance_loss = self.balance_loss_weight * aux["balance_loss"]

        loss = bpr_loss + reg_loss + balance_loss
        losses = {
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
            "balance_loss": balance_loss,
        }
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        return self._mask_predict(full_preds, train_mask)

    def _layer_usage_stat(self, layer: GraphMoELayer) -> Dict:
        expert_names = [
            f"expert_{idx}_{'1hop' if idx < layer.num_experts_1hop else '2hop'}"
            for idx in range(layer.num_experts)
        ]
        if layer._last_topk is None:
            zeros = [0 for _ in expert_names]
            return {
                "expert_names": expert_names,
                "topk": self.top_k,
                "pair_count": 0,
                "selected_count": 0,
                "selection_counts": zeros,
                "selection_percentages": [0.0 for _ in expert_names],
                "selection_token_percentages": [0.0 for _ in expert_names],
                "gate_weight_percentages": [0.0 for _ in expert_names],
            }

        topk = layer._last_topk.reshape(-1)
        counts = t.bincount(topk, minlength=layer.num_experts).float()
        selected = float(counts.sum().item())
        contexts = int(layer._last_topk.shape[0])
        percentages = (counts / max(1.0, selected) * 100.0).detach().cpu().tolist()

        if layer._last_gates is not None:
            gate_sums = layer._last_gates.sum(dim=0).float()
            gate_total = float(gate_sums.sum().item())
            gate_weight_percentages = (gate_sums / max(1.0e-12, gate_total) * 100.0).detach().cpu().tolist()
        else:
            gate_weight_percentages = [0.0 for _ in expert_names]

        return {
            "expert_names": expert_names,
            "topk": self.top_k,
            "pair_count": contexts,
            "selected_count": int(selected),
            "selection_counts": counts.long().detach().cpu().tolist(),
            "selection_percentages": percentages,
            "selection_token_percentages": percentages,
            "gate_weight_percentages": gate_weight_percentages,
        }

    def get_expert_selection_stats(self, scope: str = "all_nodes") -> Dict:
        """Compatibility hook for CoMoE's trainer.log_expert_usage."""
        layer_stats = {idx: self._layer_usage_stat(layer) for idx, layer in enumerate(self.layers)}
        if not layer_stats:
            expert_names = [f"expert_{idx}" for idx in range(self.num_experts)]
            return {
                "scope": scope,
                "expert_names": expert_names,
                "topk": self.top_k,
                "pair_count": 0,
                "selected_count": 0,
                "selection_counts": [0 for _ in expert_names],
                "selection_percentages": [0.0 for _ in expert_names],
                "selection_token_percentages": [0.0 for _ in expert_names],
                "gate_weight_percentages": [0.0 for _ in expert_names],
                "layer_stats": {},
            }

        # Aggregate layer-wise counts for the top-level summary.
        first = next(iter(layer_stats.values()))
        expert_names = first["expert_names"]
        counts = t.zeros(self.num_experts)
        gate_pcts_accum = t.zeros(self.num_experts)
        contexts = 0
        selected = 0
        for stat in layer_stats.values():
            counts += t.tensor(stat["selection_counts"], dtype=t.float32)
            gate_pcts_accum += t.tensor(stat["gate_weight_percentages"], dtype=t.float32)
            contexts += int(stat["pair_count"])
            selected += int(stat["selected_count"])

        selection_percentages = (counts / max(1.0, float(counts.sum().item())) * 100.0).tolist()
        gate_weight_percentages = (gate_pcts_accum / max(1, len(layer_stats))).tolist()
        return {
            "scope": scope,
            "expert_names": expert_names,
            "topk": self.top_k,
            "pair_count": contexts,
            "selected_count": selected,
            "selection_counts": counts.long().tolist(),
            "selection_percentages": selection_percentages,
            "selection_token_percentages": selection_percentages,
            "gate_weight_percentages": gate_weight_percentages,
            "layer_stats": layer_stats,
        }

    def analyze_moe(self) -> Dict:
        return self.get_expert_selection_stats(scope="all_nodes")
