from pytracking.parameter.segm_sk_meanmax import default_params


def parameters():
    ep_num = 21
    params = default_params.parameters()
    params.train_skipping = 3
    params.segm_net_path = '/home/slz/GitHub/d3s/save/checkpoints/ltr/segm_sk_meanmax/segm_sk_meanmax_default/SegmNet_ep00{:02d}.pth.tar'.format(
        ep_num)
    return params
