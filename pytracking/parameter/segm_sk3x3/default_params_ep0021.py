from pytracking.parameter.segm_sk3x3 import default_params


def parameters():
    ep_num = 21
    params = default_params.parameters()
    params.segm_net_path = '/home/slz/GitHub/d3s/save/checkpoints/ltr/segm_sk3x3/segm_sk3x3_default/SegmNet_ep{:04d}.pth.tar'.format(ep_num)
    return params
