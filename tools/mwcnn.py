import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2) + dilation - 1, bias=bias,
                     dilation=dilation)


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class BBlock(nn.Module):
    def __init__(self, conv, in_channels, out_channels, kernel_size, bias=True, act=nn.ReLU(True)):
        super(BBlock, self).__init__()
        m = [conv(in_channels, out_channels, kernel_size, bias=bias), act]

        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x


class DBlock(nn.Module):
    def __init__(self, conv, in_channels, out_channels, kernel_size, bias=True, act=nn.ReLU(True), one=2,
                 two=1):
        super(DBlock, self).__init__()
        m = [conv(in_channels, out_channels, kernel_size, bias=bias, dilation=one), act,
             conv(in_channels, out_channels, kernel_size, bias=bias, dilation=two), act]

        self.body = nn.Sequential(*m)

    def forward(self, x):
        x = self.body(x)
        return x
