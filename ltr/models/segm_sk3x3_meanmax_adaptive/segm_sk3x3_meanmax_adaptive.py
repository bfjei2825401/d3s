import ltr.models.backbone as backbones
import ltr.models.segm_sk3x3_meanmax_adaptive as models
from ltr import model_constructor
from ltr.models.segm.segm import SegmNet


@model_constructor
def segm_sk3x3_meanmax_adaptive_resnet18(segm_input_dim=(256, 256), segm_inter_dim=(256, 256), backbone_pretrained=True,
                                         topk_pos=3,
                                         topk_neg=3, mixer_channels=2, scale_num=2):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 64, 128, 256)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = models.SegmNetSK3x3MeanMaxAdaptive(segm_input_dim=segm_input_dim,
                                                        segm_inter_dim=segm_inter_dim,
                                                        segm_dim=segm_dim,
                                                        topk_pos=topk_pos,
                                                        topk_neg=topk_neg,
                                                        mixer_channels=mixer_channels,
                                                        scale_num=scale_num)

    net = SegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                  segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)

    return net


@model_constructor
def segm_sk3x3_meanmax_adaptive_resnet50(segm_input_dim=(256, 256), segm_inter_dim=(256, 256), backbone_pretrained=True,
                                         topk_pos=3,
                                         topk_neg=3, mixer_channels=2, scale_num=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = models.SegmNetSK3x3MeanMaxAdaptive(segm_input_dim=segm_input_dim,
                                                        segm_inter_dim=segm_inter_dim,
                                                        segm_dim=segm_dim,
                                                        topk_pos=topk_pos,
                                                        topk_neg=topk_neg,
                                                        mixer_channels=mixer_channels,
                                                        scale_num=scale_num)

    net = SegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                  segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net
