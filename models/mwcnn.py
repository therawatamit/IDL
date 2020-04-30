import torch.nn as nn
from tools.mwcnn import DWT, IWT, BBlock, DBlock, default_conv


class MWCNN(nn.Module):
    def __init__(self, conv=default_conv):
        super(MWCNN, self).__init__()
        n_feats = 64
        kernel_size = 3
        nColor = 1
        self.DWT = DWT()
        self.IWT = IWT()

        m_head = [BBlock(conv, nColor, n_feats, kernel_size)]

        d_l0 = [DBlock(conv, n_feats, n_feats, kernel_size)]

        d_l1 = [BBlock(conv, n_feats * 4, n_feats * 2, kernel_size),
                DBlock(conv, n_feats * 2, n_feats * 2, kernel_size)]

        d_l2 = [BBlock(conv, n_feats * 8, n_feats * 4, kernel_size),
                DBlock(conv, n_feats * 4, n_feats * 4, kernel_size)]

        pro_l3 = [BBlock(conv, n_feats * 16, n_feats * 8, kernel_size),
                  DBlock(conv, n_feats * 8, n_feats * 8, kernel_size, two=3),
                  DBlock(conv, n_feats * 8, n_feats * 8, kernel_size, one=3, two=2),
                  BBlock(conv, n_feats * 8, n_feats * 16, kernel_size)]

        i_l2 = [DBlock(conv, n_feats * 4, n_feats * 4, kernel_size),
                BBlock(conv, n_feats * 4, n_feats * 8, kernel_size)]

        i_l1 = [DBlock(conv, n_feats * 2, n_feats * 2, kernel_size),
                BBlock(conv, n_feats * 2, n_feats * 4, kernel_size)]

        i_l0 = [DBlock(conv, n_feats, n_feats, kernel_size)]

        m_tail = [conv(n_feats, nColor, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x0 = self.d_l0(self.head(x))
        x1 = self.d_l1(self.DWT(x0))
        x2 = self.d_l2(self.DWT(x1))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        x_ = self.IWT(self.i_l1(x_)) + x0
        x = self.tail(self.i_l0(x_)) + x

        return x
