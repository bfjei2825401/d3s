from pytracking.parameter.segm_se import finetune_params


def parameters(ep_num):
    params = finetune_params.parameters()
    params.segm_net_path = '/home/slz/GitHub/d3s/save/checkpoints/ltr/segm_se/segm_se_finetune/SegmNet_ep00{:0>2d}.pth.tar'.format(ep_num)
    return params
