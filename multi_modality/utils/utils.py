from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

@torch.no_grad()
def flops(model, size, *kwargs, round_num=1, eval=True, device="cuda", fp16=False, ):
    if eval: model.eval()
    with torch.cuda.amp.autocast(enabled=fp16):
        inputs = torch.randn(size, device=device, requires_grad=True)
        with torch.no_grad():
            flops = FlopCountAnalysis(model, (inputs, *kwargs))
            flops_num = flops.total() / 1000000000

    print(flop_count_table(flops))
    print(f"fvcore flops : {flops_num}")

    return round(flops_num, round_num)