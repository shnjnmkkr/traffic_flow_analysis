import torch
import torch.nn as nn

class DCAE(nn.Module):
    """
    Deep Convolutional Autoencoder (DCAE) for feature extraction and classification.
    Encoder outputs a feature vector; decoder reconstructs the input.
    Classification head predicts vehicle type.
    """
    def __init__(self, num_classes=3, feature_dim=128):
        super(DCAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 32x32x32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x16x16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128x8x8
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128*8*8, feature_dim),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder_fc = nn.Linear(feature_dim, 128*8*8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 64x16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 32x32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  # 3x64x64
            nn.Sigmoid()
        )
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x: (batch, 3, 64, 64)
        feat = self.encoder(x)
        recon = self.decoder(self.decoder_fc(feat))
        logits = self.classifier(feat)
        return feat, recon, logits

# Example usage:
# model = DCAE(num_classes=3, feature_dim=128)
# feat, recon, logits = model(torch.randn(1, 3, 64, 64)) 