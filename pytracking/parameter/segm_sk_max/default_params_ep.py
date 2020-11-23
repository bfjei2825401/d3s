from pytracking.parameter.segm_sk3x3_max import default_params


def parameters(ep_num):
    params = default_params.parameters()
    params.segm_net_path = '/home/slz/GitHub/d3s/save/checkpoints/ltr/segm_sk_max/segm_sk_max_default/SegmNet_ep00{:0>2d}.pth.tar'.format(
        ep_num)
    return params
