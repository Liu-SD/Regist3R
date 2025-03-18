# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# global alignment optimization wrapper function
# --------------------------------------------------------
from regist3r.cloud_opt.direct_register import DirectRegister


def global_aligner(regist3r_output, device):
    # extract all inputs
    view, pred, index, ref_index = [regist3r_output[k] for k in 'view pred index ref_index'.split()]
    # build the optimizer
    net = DirectRegister(view, pred, index, ref_index).to(device)
    return net
