from torchvision import models


def lraspp_mobilenet_v3_large(outputchannels=1, keep_feature_extract=False, use_pretrained=True):
    """ lraspp pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    """
    model_lraspp = models.segmentation.lraspp_mobilenet_v3_large(pretrained=use_pretrained, progress=True)
    model_lraspp.aux_classifier = None
    if keep_feature_extract:
        for param in model_lraspp.parameters():
            param.requires_grad = False

    model_lraspp.classifier = models.segmentation.lraspp.LRASPPHead(low_channels=40, high_channels=960, num_classes=outputchannels, inter_channels=128)

    return model_lraspp