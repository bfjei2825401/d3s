import os
from pytracking.parameter.segm_sk_meanmax import default_params


def parameters(ep_num):
    params = default_params.parameters()
    params.filename = os.path.basename(__file__).split('.')[0]
    params.tracker_name = 'segm_sk3x3_meanmax_adaptive'
    params.segm_net_path = '/home/slz/GitHub/d3s/save/checkpoints/ltr/segm_sk3x3_meanmax_adaptive/segm_sk3x3_meanmax_adaptive_default/SegmNet_ep00{:0>2d}.pth.tar'.format(
        ep_num)
    return params
