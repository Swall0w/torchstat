import torch.nn as nn
import torch
import numpy as np


def compute_memory(module, inp, out):
    if isinstance(module, nn.ReLU):
        return compute_ReLU_memory(module, inp, out)
    else:
        print(f"[Memory]: {type(module).__name__} is not supported!")
        return (0, 0)
    pass


def compute_ReLU_memory(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU))
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * inp.size()[1:].numel()

    return (mread, mwrite)


# def compute_Conv2d_flops(module, inp, out):
#    # Can have multiple inputs, getting the first one
#    assert isinstance(module, nn.Conv2d)
#    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
#
#    batch_size = inp.size()[0]
#    in_c = inp.size()[1]
#    k_h, k_w = module.kernel_size
#    out_c, out_h, out_w = out.size()[1:]
#    groups = module.groups
#
#    filters_per_channel = out_c // groups
#    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
#    active_elements_count = batch_size * out_h * out_w
#
#    total_conv_flops = conv_per_position_flops * active_elements_count
#
#    bias_flops = 0
#    if module.bias is not None:
#        bias_flops = out_c * active_elements_count
#
#    total_flops = total_conv_flops + bias_flops
#    return total_flops
