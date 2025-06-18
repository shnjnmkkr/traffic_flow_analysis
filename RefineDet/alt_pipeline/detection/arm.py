import torch
import torch.nn as nn

class AnchorRefinementModule(nn.Module):
    """
    Anchor Refinement Module (ARM) for RefineDet.
    Predicts anchor adjustments (deltas) and objectness scores from feature maps.
    """
    def __init__(self, in_channels, num_anchors=6):
        super(AnchorRefinementModule, self).__init__()
        # Predict anchor deltas (dx, dy, dw, dh) for each anchor
        self.loc = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        # Predict objectness score for each anchor
        self.conf = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=3, padding=1)  # 2: object / not object

    def forward(self, x):
        # x: (batch, in_channels, H, W)
        loc = self.loc(x)  # (batch, num_anchors*4, H, W)
        conf = self.conf(x)  # (batch, num_anchors*2, H, W)
        return loc, conf

# Example usage:
# arm = AnchorRefinementModule(in_channels=512, num_anchors=6)
# loc, conf = arm(torch.randn(1, 512, 38, 38)) 