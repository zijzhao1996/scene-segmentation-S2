from torchvision import models


def deeplabv3_mobilenet_v3_large(outputchannels=1, keep_feature_extract=False, use_pretrained=True):
    """ DeepLabV3 pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    """
    model_deeplabv3 = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=use_pretrained, progress=True)
    model_deeplabv3.aux_classifier = None
    if keep_feature_extract:
        for param in model_deeplabv3.parameters():
            param.requires_grad = False

    model_deeplabv3.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, outputchannels)

    return model_deeplabv3
