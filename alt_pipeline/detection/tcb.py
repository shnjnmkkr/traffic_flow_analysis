import torch
import torch.nn as nn

class TransferConnectionBlock(nn.Module):
    """
    Transfer Connection Block (TCB) for RefineDet.
    Refines and transfers features from ARM to ODM.
    """
    def __init__(self, in_channels):
        super(TransferConnectionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, up=None):
        # x: feature from ARM
        # up: upsampled feature from deeper TCB (or None for the deepest layer)
        out = self.conv1(x)
        out = self.relu(out)
        if up is not None:
            out = out + self.upsample(up)
        out = self.conv2(out)
        out = self.relu(out)
        return out

# Example usage:
# tcb = TransferConnectionBlock(in_channels=512)
# out = tcb(torch.randn(1, 512, 38, 38), up=None) 