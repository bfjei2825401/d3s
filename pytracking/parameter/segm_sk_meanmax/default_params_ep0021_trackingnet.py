from pytracking.parameter.segm_sk_meanmax import default_params


def parameters():
    ep_num = 21
    params = default_params.parameters()
    # params.max_image_sample_size = (18 * 16) ** 2
    # params.search_area_scale = 5  # Scale relative to target size
    params.segm_net_path = '/home/slz/GitHub/d3s/save/checkpoints/ltr/segm_sk_meanmax/segm_sk_meanmax_default/SegmNet_ep00{:0>2d}.pth.tar'.format(
        ep_num)
    return params
