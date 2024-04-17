# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import torch
from einops import rearrange
from typing import Callable, Tuple, List, Union

def get_objective_score(score_attn):
    score_attn = score_attn.mean(dim=1)
    scores = (score_attn * torch.log(score_attn)).sum(dim=2).unsqueeze(-1)

    # BACKGROUND REMOVING
    B, T_R, _ = scores.shape
    scores = scores - scores.amin(dim=1, keepdim=True)
    scores = scores / scores.amax(dim=1, keepdim=True)
    score_mask = scores < scores.mean(dim=1, keepdim=True)

    # FOREGROUND SHARPENING
    scores = scores - scores.mean(dim=1, keepdim=True)
    scores = scores / scores.amax(dim=1, keepdim=True)
    scores[score_mask] = 0.0

    return scores



def vidTLDR(x, attn, info, layer):
    if not info["use"]:
        return x

    B, T, _ = x.shape
    r = info["r"][layer]
    r_merge = int(T * r) if r < 1 else r
    r_merge = max(min(r_merge, T // 2, T), 0)
    if not r_merge:
        return x

    if info["source_trace"] and info["source"] is None:
        info["source"] = torch.eye(T, device=x.device)[None, ...].expand(B, T, T) # (1, 1752, 1752)

    score_obj = get_objective_score(attn) if info["use"] else None

    merge = merging(
        x,
        r_merge        = r_merge,
        score_obj      = score_obj,
    )

    x, info["size"], info["source"] \
        = merge_wavg(merge, x, info["size"],
                     source_trace     = info["source_trace"],
                     source           = info["source"])

    return x


def merging(
    metric: torch.Tensor,
    r_merge        : int,
    score_obj      : torch.Tensor,
    ):

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # (1, 2352, 768)

        # SECTION I. TOKEN MERGING
        a, b = metric[..., ::2, :], metric[..., 1::2, :]  # (12, 99, 64), (12, 98, 64)
        n, s, t1, t2 = a.shape[0], a.shape[1], a.shape[-2], b.shape[-2]

        scores = (a @ b.transpose(-1, -2) + 1) / 2 # 0 - 1

        # TOKEN MERGING
        node_max, node_idx = scores.max(dim=-1) # (12, 99), (12, 99)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # (12, 99, 1)
        unm_idx  = edge_idx[..., r_merge:, :]  # Unmerged Tokens (12, 83, 1)
        src_idx  = edge_idx[..., :r_merge, :]  # Merged Tokens   (12, 16, 1)
        dst_idx  = node_idx[..., None].gather(dim=-2, index=src_idx)  # (12, 16, 1)
        unm_idx  = unm_idx.sort(dim=1)[0]

        src_so = None
        if score_obj is not None:
            src_so, dst_so = score_obj[..., ::2, :], score_obj[..., 1::2, :] # (1, 1176, 1)
            src_so = src_so.gather(dim=-2, index=src_idx)  # (12, 91, 197)

    def merge(x: torch.Tensor, mode = "sum", dtype = torch.float32):
        ori_dtype = x.dtype
        x = x.to(dtype=dtype)
        src, dst = x[..., ::2, :], x[..., 1::2, :] # (12, 99, 197), (12, 98, 197)
        n, mid, c = src.shape[0], src.shape[1:-2], src.shape[-1]
        unm = src.gather(dim=-2, index=unm_idx.expand(n, *mid, t1 - r_merge, c)) # (12, 91, 197)
        src = src.gather(dim=-2, index=src_idx.expand(n, *mid, r_merge, c))
        if score_obj is not None:
            src = src * src_so
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, *mid, r_merge, c), src, reduce=mode)  # (12, 98, 197)
        x = torch.cat([unm, dst], dim=-2)  # (12, 1 + 180, 197)
        x = x.to(dtype=ori_dtype)
        return x

    return merge

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor,
    source_trace: int = 0, source: list = None):
    if size == None: size = torch.ones_like(x[..., 0, None])

    size_max = size.amax(dim=-2, keepdim=True)
    x = merge(x * (size / size_max), mode="sum")
    size = merge(size, mode="sum")
    x = x / (size / size_max)

    if source_trace:
        source = merge(source, mode="amax")

    return x, size, source
