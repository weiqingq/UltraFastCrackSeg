import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm = nn.GroupNorm(4, out_channels)
        self.drop_path = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x):
        return self.drop_path(F.gelu(self.norm(self.conv(x))))


class UltraFastCrackSeg(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[32, 64, 72, 96, 128], drop_path_rate=0.1, pretrained_path=None):
        super().__init__()
        self.c_list = c_list

        # Compute drop path rates
        depths = [1] * len(c_list)  # Assuming each stage has one block
        encoder_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        decoder_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) - 1)][::-1]

        # Define encoder modules using only ConvBlock
        self.encoders = nn.ModuleList()
        for i, (in_ch, out_ch, dp_rate) in enumerate(zip([input_channels] + c_list[:-1], c_list, encoder_dp_rates)):
            self.encoders.append(ConvBlock(in_ch, out_ch, dp_rate))

        # Define decoder modules using only ConvBlock
        self.decoders = nn.ModuleList()
        for i, (in_ch, out_ch, dp_rate) in enumerate(zip(c_list[::-1][:len(c_list)-1], c_list[::-1][1:], decoder_dp_rates)):
            self.decoders.append(ConvBlock(in_ch, out_ch, dp_rate))

        self.final_conv = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.apply(self._init_weights)

        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for encoder in self.encoders:
            # import pdb; pdb.set_trace()
            x = F.max_pool2d(encoder(x), 2, 2)
            skip_connections.append(x)

        # Reverse to match the order for decoder processing
        skip_connections = skip_connections[::-1]

        # Decoder path with skip connections
        for i, decoder in enumerate(self.decoders):
            x = F.gelu(F.interpolate(decoder(x), scale_factor=2, mode='bilinear', align_corners=True))
            x = x + skip_connections[i+1]

        x = self.final_conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))
        return torch.sigmoid(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_pretrained_weights(self, pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
        # Load state dict to the model, capturing missing and unexpected keys
        load_result = self.load_state_dict(checkpoint, strict=False)
        if load_result.missing_keys:
            print("Missing keys in state_dict:", load_result.missing_keys)
        if load_result.unexpected_keys:
            print("Unexpected keys in state_dict:", load_result.unexpected_keys)
        print('weight loaded')
