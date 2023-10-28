import torch.nn as nn
import torch

class UNet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UNet, self).__init__()

    # First encoding block
    self.enc1 = self.conv_block(in_channels, 64)
    self.pool1 = nn.MaxPool2d(kernel_size=2)

    # Second encoding block
    self.enc2 = self.conv_block(64, 128)
    self.pool2 = nn.MaxPool2d(kernel_size=2)

    # Third encoding block
    self.enc3 = self.conv_block(128, 256)
    self.pool3 = nn.MaxPool2d(kernel_size=2)

    # Fourth encoding block
    self.enc4 = self.conv_block(256, 512)
    self.pool4 = nn.MaxPool2d(kernel_size=2)

    # Bottleneck
    self.bottleneck = self.conv_block(512, 1024)

    # Expansive Path
    # First decoding block
    self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.dec4 = self.conv_block(1024, 512)

    # Second decoding block
    self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.dec3 = self.conv_block(512, 256)

    # Third decoding block
    self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.dec2 = self.conv_block(256, 128)

    # Fourth decoding block
    self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.dec1 = self.conv_block(128, 64)

    # Final output
    self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

  def forward(self, x):
    # Contracting Path
    enc1 = self.enc1(x)
    enc1_pool = self.pool1(enc1)

    enc2 = self.enc2(enc1_pool)
    enc2_pool = self.pool2(enc2)

    enc3 = self.enc3(enc2_pool)
    enc3_pool = self.pool3(enc3)

    enc4 = self.enc4(enc3_pool)
    enc4_pool = self.pool4(enc4)

    # Bottleneck
    bottleneck = self.bottleneck(enc4_pool)

    # Expansive Path
    upconv4 = self.upconv4(bottleneck)
    merge4 = torch.cat([upconv4, enc4], dim=1)
    dec4 = self.dec4(merge4)

    upconv3 = self.upconv3(dec4)
    merge3 = torch.cat([upconv3, enc3], dim=1)
    dec3 = self.dec3(merge3)

    upconv2 = self.upconv2(dec3)
    merge2 = torch.cat([upconv2, enc2], dim=1)
    dec2 = self.dec2(merge2)

    upconv1 = self.upconv1(dec2)
    merge1 = torch.cat([upconv1, enc1], dim=1)
    dec1 = self.dec1(merge1)

    return self.outconv(dec1)

  def conv_block(self, in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return block