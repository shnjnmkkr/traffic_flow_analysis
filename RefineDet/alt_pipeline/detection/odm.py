import torch
import torch.nn as nn

class ObjectDetectionModule(nn.Module):
    """
    Object Detection Module (ODM) for RefineDet.
    Predicts final class scores and bounding box regressions from TCB-refined features.
    """
    def __init__(self, in_channels, num_anchors=6, num_classes=2):
        super(ObjectDetectionModule, self).__init__()
        # Predict bounding box deltas (dx, dy, dw, dh) for each anchor
        self.loc = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        # Predict class scores for each anchor
        self.conf = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (batch, in_channels, H, W)
        loc = self.loc(x)  # (batch, num_anchors*4, H, W)
        conf = self.conf(x)  # (batch, num_anchors*num_classes, H, W)
        return loc, conf

# Example usage:
# odm = ObjectDetectionModule(in_channels=512, num_anchors=6, num_classes=2)
# loc, conf = odm(torch.randn(1, 512, 38, 38)) 