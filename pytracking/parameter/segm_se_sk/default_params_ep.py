from pytracking.parameter.segm_se_sk import default_params


def parameters(ep_num):
    params = default_params.parameters()
    params.segm_net_path = '/home/slz/GitHub/d3s/save/checkpoints/ltr/segm_se_sk/segm_se_sk_default/SegmNet_ep00{:0>2d}.pth.tar'.format(ep_num)
    return params
