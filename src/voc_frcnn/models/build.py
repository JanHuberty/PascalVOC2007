from __future__ import annotations

import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes: int = 21) -> nn.Module:
    """
    Builds Faster R-CNN with a ResNet-50 FPN backbone.
    num_classes includes background, so 21 for VOC (20 + background).
    We avoid downloading weights by default for reproducibility.
    """
    model = fasterrcnn_resnet50_fpn(weights=None)  # no internet dependency
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
