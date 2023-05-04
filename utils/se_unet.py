import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, reduction_ratio):
        super(UNetDec, self).__init__()

        self.up = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio is not None:
            self.excitation1 = nn.Conv3d(out_channels, out_channels // self.reduction_ratio, kernel_size=1)
            self.excitation2 = nn.Conv3d(out_channels // self.reduction_ratio, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.up(x)
        if self.reduction_ratio:
            fm_size = out.size()[2:]
            scale_weight = F.avg_pool3d(out, fm_size)
            scale_weight = F.relu(self.excitation1(scale_weight))
            scale_weight = F.sigmoid(self.excitation2(scale_weight))
            out = out * scale_weight.expand_as(out)
        return out


class UNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, reduction_ratio, dropout=False):
        super(UNetEnc, self).__init__()

        layers = [
            nn.Conv3d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, stride=2, ceil_mode=True)

        ]
        if dropout:
            layers += [nn.Dropout(.5)]

        self.down = nn.Sequential(*layers)
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio is not None:
            self.excitation1 = nn.Conv3d(out_channels, out_channels // self.reduction_ratio, kernel_size=1)
            self.excitation2 = nn.Conv3d(out_channels // self.reduction_ratio, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.down(x)
        if self.reduction_ratio:
            fm_size = out.size()[2:]
            scale_weight = F.avg_pool3d(out, fm_size)
            scale_weight = F.relu(self.excitation1(scale_weight))
            scale_weight = F.sigmoid(self.excitation2(scale_weight))
            out = out * scale_weight.expand_as(out)
        return out


class Bottleneck_block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, reduction_ratio, dropout=False):
        super(Bottleneck_block, self).__init__()

        layers = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)]

        if dropout:
            layers += [nn.Dropout(.5)]

        self.center = nn.Sequential(*layers)
        self.reduction_ratio = reduction_ratio
        if self.reduction_ratio is not None:
            self.excitation1 = nn.Conv3d(out_channels, out_channels // self.reduction_ratio, kernel_size=1)
            self.excitation2 = nn.Conv3d(out_channels // self.reduction_ratio, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.center(x)
        if self.reduction_ratio:
            fm_size = out.size()[2:]
            scale_weight = F.avg_pool3d(out, fm_size)
            scale_weight = F.relu(self.excitation1(scale_weight))
            scale_weight = F.sigmoid(self.excitation2(scale_weight))
            out = out * scale_weight.expand_as(out)
        return out


class UNet(nn.Module):

    def __init__(self, input_channels, num_classes, init_features, network_depth, bottleneck_layers, reduction_ratio):
        super(UNet, self).__init__()

        self.reduction_ratio = reduction_ratio
        self.network_depth = network_depth
        self.bottleneck_layers = bottleneck_layers
        skip_connection_channel_counts = []

        self.add_module('firstconv', nn.Conv3d(in_channels=input_channels,
                                               out_channels=init_features, kernel_size=3,
                                               stride=1, padding=1, bias=True))

        self.encodingBlocks = nn.ModuleList([])
        features = init_features
        for i in range(self.network_depth):
            self.encodingBlocks.append(UNetEnc(features, 2*features, reduction_ratio=self.reduction_ratio))
            skip_connection_channel_counts.insert(0, 2*features)
            features *= 2
        final_encoding_channels = skip_connection_channel_counts[0]

        self.bottleNecks = nn.ModuleList([])
        for i in range(self.bottleneck_layers):
            dilation_factor = 1
            self.bottleNecks.append(Bottleneck_block(final_encoding_channels,
                                                     final_encoding_channels, dilation_rate=dilation_factor,
                                                     reduction_ratio=self.reduction_ratio))

        self.decodingBlocks = nn.ModuleList([])
        for i in range(self.network_depth):
            if i == 0:
                prev_deconv_channels = final_encoding_channels
            self.decodingBlocks.append(UNetDec(prev_deconv_channels + skip_connection_channel_counts[i],
                                               skip_connection_channel_counts[i],
                                               reduction_ratio=self.reduction_ratio))
            prev_deconv_channels = skip_connection_channel_counts[i]

        self.final = nn.Conv3d(2 * init_features, num_classes, 1)

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        out = self.firstconv(x)

        skip_connections = []
        for i in range(self.network_depth):
            out = self.encodingBlocks[i](out)
            skip_connections.append(out)
        for i in range(self.bottleneck_layers):
            out = self.bottleNecks[i](out)

        for i in range(self.network_depth):
            skip = skip_connections.pop()
            out = self.decodingBlocks[i](torch.cat([out, skip], 1))

        out = self.final(out)
        return out.permute(0, 1, 3, 4, 2).contiguous()
